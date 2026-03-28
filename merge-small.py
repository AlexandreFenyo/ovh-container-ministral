import argparse
import json
import os
import shlex
import shutil

import torch
from huggingface_hub import hf_hub_download, save_torch_model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _coerce_value(raw):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _load_params(path):
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = shlex.split(stripped)
            if not tokens or "=" not in tokens[0]:
                continue
            key, first_value = tokens[0].split("=", 1)
            values = [first_value] + tokens[1:]
            if len(values) == 1:
                params[key] = _coerce_value(values[0])
            else:
                params[key] = [_coerce_value(v) for v in values]
    return params


def _get_param(params, key, default=None, required=True):
    if key in params:
        return params[key]
    if required:
        raise ValueError(f"Missing required parameter: {key}")
    return default


def _load_tokenizer(tokenizer_name):
    attempts = [
        {
            "kwargs": {"use_fast": True, "fix_mistral_regex": True},
            "reason": "preferred tokenizer load with mistral regex fix",
        },
        {
            "kwargs": {"use_fast": True, "fix_mistral_regex": False},
            "reason": "fallback when the local tokenizers API is incompatible with the mistral regex patch",
        },
        {
            "kwargs": {"use_fast": False},
            "reason": "last-resort tokenizer load without fast-tokenizer features",
        },
    ]

    errors = []
    for attempt in attempts:
        try:
            if errors:
                print(
                    f"Retrying tokenizer load for {tokenizer_name!r}: {attempt['reason']}."
                )
            return AutoTokenizer.from_pretrained(tokenizer_name, **attempt["kwargs"])
        except Exception as exc:
            errors.append((attempt["kwargs"], exc))
            message = str(exc)
            can_retry = False
            if (
                attempt["kwargs"].get("fix_mistral_regex") is True
                and "backend_tokenizer" in message
            ):
                can_retry = True
            elif "fix_mistral_regex" in message:
                can_retry = True

            if not can_retry:
                raise

    attempted = ", ".join(str(kwargs) for kwargs, _ in errors)
    raise RuntimeError(
        f"Unable to load tokenizer {tokenizer_name!r} after trying: {attempted}"
    ) from errors[-1][1]


def _copy_local_file_if_present(source_dir, filename, out_dir):
    source_path = os.path.join(source_dir, filename)
    if not os.path.exists(source_path):
        return False
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(source_path, os.path.join(out_dir, filename))
    return True


def _maybe_copy_file_from_hub(repo_id, filename, out_dir):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return False
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(out_dir, filename))
    return True


def _rewrite_config_for_ollama(out_dir, dtype_name):
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["model_type"] = "mistral"
    config["architectures"] = ["MistralForCausalLM"]
    config["torch_dtype"] = dtype_name
    config.pop("dtype", None)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_ollama_modelfile(out_dir, modelfile_path):
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(f"FROM {os.path.abspath(out_dir)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", type=str, default="./ministral-8b-instruct-merged"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_shard_size", type=str, default="2GB")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Chemin de l'adaptateur LoRA a fusionner. Si omis, utilise ft_output_dir du fichier de params.",
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default=None,
        help="Nom ou chemin du modele base. Si omis, utilise model_name du fichier de params.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Nom ou chemin du tokenizer. Si omis, utilise tokenizer du fichier de params.",
    )
    parser.add_argument(
        "--ollama-compatible",
        action="store_true",
        help="Exporte un modele fusionne plus facile a importer dans Ollama: force float16 si necessaire et normalise config.json vers Mistral.",
    )
    parser.add_argument(
        "--write-ollama-modelfile",
        type=str,
        default=None,
        help="Ecrit un Modelfile Ollama pointant vers out_dir.",
    )
    args = parser.parse_args()

    params_path = os.environ.get("PARAMS_CFG", "params-small.cfg")
    if not os.path.isabs(params_path):
        params_path = os.path.join(os.path.dirname(__file__), params_path)
    params = _load_params(params_path)

    adapter_path = args.adapter_path or _get_param(params, "ft_output_dir")
    base_model_name = args.base_model_name or _get_param(
        params, "model_name", required=False
    ) or _get_param(params, "tokenizer")
    tokenizer_name = args.tokenizer_name or _get_param(params, "tokenizer")
    attn_implementation = _get_param(
        params, "attn_implementation", default="eager", required=False
    )

    dtype_name = args.dtype
    if args.ollama_compatible and dtype_name == "bfloat16":
        print("Ollama-compatible export requested: forcing dtype=float16 instead of bfloat16.")
        dtype_name = "float16"
    dtype = getattr(torch, dtype_name)

    tokenizer = _load_tokenizer(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    model.config.save_pretrained(args.out_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    tied = getattr(model, "_tied_weights_keys", None)
    save_torch_model(
        model,
        args.out_dir,
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
        shared_tensors_to_discard=tied,
    )

    if not _copy_local_file_if_present(adapter_path, "chat_template.jinja", args.out_dir):
        _maybe_copy_file_from_hub(adapter_path, "chat_template.jinja", args.out_dir)

    if args.ollama_compatible:
        _rewrite_config_for_ollama(args.out_dir, dtype_name)

    if args.write_ollama_modelfile:
        _write_ollama_modelfile(args.out_dir, args.write_ollama_modelfile)

    print(f"OK: modele merge sauvegarde dans: {args.out_dir}")
    if args.ollama_compatible:
        print("Config Ollama-compatible ecrite: model_type=mistral.")
    if args.write_ollama_modelfile:
        print(f"Modelfile Ollama ecrit dans: {args.write_ollama_modelfile}")


if __name__ == "__main__":
    main()
