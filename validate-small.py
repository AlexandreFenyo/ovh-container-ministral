import argparse
import os
import shlex
import sys
import unicodedata

import torch
from datasets import load_dataset
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


def _discover_variants(adapter_root):
    variants = []
    if not os.path.isdir(adapter_root):
        return variants

    checkpoint_variants = []
    for entry in os.listdir(adapter_root):
        if not entry.startswith("checkpoint-"):
            continue
        path = os.path.join(adapter_root, entry)
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            continue
        try:
            step = int(entry.split("-", 1)[1])
        except ValueError:
            continue
        checkpoint_variants.append((step, entry, path))

    for _, entry, path in sorted(checkpoint_variants):
        variants.append((entry, path))

    final_adapter_path = os.path.join(adapter_root, "adapter_config.json")
    if os.path.isfile(final_adapter_path):
        variants.append(("final", adapter_root))

    return variants


def _normalize_text(text):
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(normalized.split())
    return normalized.strip()


def _build_prompt(messages):
    if not messages:
        raise ValueError("Cannot build a prompt from an empty message list.")

    prompt_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        prompt_messages.append(
            {
                "role": message["role"],
                "content": message.get("content", ""),
            }
        )
    return prompt_messages


def _generate_response(model, tokenizer, messages, max_new_tokens):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def _load_base_model(base_model_name, model_kwargs):
    return AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)


def _load_model_bundle(base_model_name, adapter_root, model_kwargs):
    variants = _discover_variants(adapter_root)
    base_model = _load_base_model(base_model_name, model_kwargs)
    adapter_variants = [(label, path) for label, path in variants]
    if not adapter_variants:
        return base_model, variants

    first_label, first_path = adapter_variants[0]
    model = PeftModel.from_pretrained(
        base_model,
        first_path,
        adapter_name=first_label,
        is_trainable=False,
    )
    for label, path in adapter_variants[1:]:
        model.load_adapter(path, adapter_name=label, is_trainable=False)
    return model, variants


def _select_split(dataset_name, requested_split):
    dataset_dict = load_dataset(dataset_name)
    available_splits = list(dataset_dict.keys())

    if requested_split:
        if requested_split not in dataset_dict:
            raise ValueError(
                f"Unknown split {requested_split!r}. Available splits: {', '.join(available_splits)}"
            )
        return requested_split, dataset_dict[requested_split], available_splits

    print(f"Available splits for {dataset_name}:")
    for split_name in available_splits:
        print(f"- {split_name}: {len(dataset_dict[split_name])} rows")

    while True:
        chosen = input("Choose a split to validate: ").strip()
        if chosen in dataset_dict:
            return chosen, dataset_dict[chosen], available_splits
        print(f"Unknown split {chosen!r}. Valid options: {', '.join(available_splits)}")


def _format_result_line(index, total, question, expected, got, ok):
    status = "OK" if ok else "FAIL"
    return (
        f"[{status}] {index + 1}/{total} | {question}\n"
        f"  expected: {expected}\n"
        f"  got     : {got}"
    )


def _enable_line_buffering():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)


def main():
    _enable_line_buffering()

    parser = argparse.ArgumentParser(
        description=(
            "Valide le finetuning en comparant les reponses generees par chaque checkpoint "
            "et le modele final sur les exemples de type negative."
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split du dataset a analyser. Si omis, une selection interactive est proposee.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Nom du dataset Hugging Face. Par defaut, celui du fichier de parametres.",
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default=None,
        help="Nom ou chemin du modele de base. Si omis, utilise le fichier de parametres.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Nom ou chemin du tokenizer. Si omis, utilise le fichier de parametres.",
    )
    parser.add_argument(
        "--adapter-root",
        type=str,
        default=None,
        help="Repertoire racine contenant les checkpoints LoRA. Si omis, utilise ft_output_dir du fichier de parametres.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Nombre maximum de tokens a generer pour chaque exemple.",
    )
    parser.add_argument(
        "--max-negative-examples",
        type=int,
        default=None,
        help=(
            "Limite le nombre d'exemples negative a analyser. "
            "Utile pour valider rapidement le script."
        ),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map utilise pour charger le modele.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precision du modele de base.",
    )
    args = parser.parse_args()

    params_path = os.environ.get("PARAMS_CFG", "params-small.cfg")
    if not os.path.isabs(params_path):
        params_path = os.path.join(os.path.dirname(__file__), params_path)
    params = _load_params(params_path)

    dataset_name = args.dataset_name or _get_param(params, "var_dataset_name")
    base_model_name = args.base_model_name or _get_param(
        params, "model_name", required=False
    ) or _get_param(params, "tokenizer")
    tokenizer_name = args.tokenizer_name or _get_param(params, "tokenizer")
    adapter_root = args.adapter_root or _get_param(params, "ft_output_dir")
    attn_implementation = _get_param(
        params, "attn_implementation", default="eager", required=False
    )
    system_prompt = _get_param(params, "system_prompt", required=False)

    split_name, dataset_split, available_splits = _select_split(dataset_name, args.split)
    negative_dataset = dataset_split.filter(lambda row: row["type"] == "negative")
    if args.max_negative_examples is not None:
        if args.max_negative_examples <= 0:
            raise ValueError("--max-negative-examples must be a positive integer")
        negative_dataset = negative_dataset.select(
            range(min(args.max_negative_examples, len(negative_dataset)))
        )

    if len(negative_dataset) == 0:
        raise RuntimeError(f"No negative examples found in split {split_name!r}.")

    tokenizer = _load_tokenizer(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.dtype)
    model_kwargs = dict(
        attn_implementation=attn_implementation,
        dtype=dtype,
        use_cache=True,
        device_map=args.device_map,
    )

    model, variants = _load_model_bundle(base_model_name, adapter_root, model_kwargs)
    if not variants:
        raise RuntimeError(
            f"No checkpoint or final adapter found in {os.path.abspath(adapter_root)}."
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Selected split: {split_name}", flush=True)
    print(f"Negative examples: {len(negative_dataset)} / {len(dataset_split)}", flush=True)
    if system_prompt:
        print("System prompt loaded from params file.", flush=True)
    print("Adapters under test:", flush=True)
    for label, path in variants:
        print(f"- {label}: {path}", flush=True)

    results = {}
    for label, _ in variants:
        if hasattr(model, "set_adapter"):
            model.set_adapter(label)

        correct = 0
        failures = []
        total = len(negative_dataset)
        for idx, row in enumerate(negative_dataset):
            prefix = f"[split={split_name}][checkpoint={label}] {idx + 1}/{total}"
            messages = row.get("messages") or []
            prompt_messages = _build_prompt(messages)
            expected = row["assistant"]
            generated = _generate_response(
                model, tokenizer, prompt_messages, args.max_new_tokens
            )

            ok = _normalize_text(generated) == _normalize_text(expected)
            print(f"{prefix} -> {'OK' if ok else 'FAIL'}", flush=True)
            if ok:
                correct += 1
            else:
                failures.append(
                    {
                        "index": idx,
                        "question": row["user"],
                        "expected": expected,
                        "got": generated,
                    }
                )

        results[label] = {
            "correct": correct,
            "total": len(negative_dataset),
            "failures": failures,
        }

    print(flush=True)
    for label, summary in results.items():
        correct = summary["correct"]
        total = summary["total"]
        accuracy = correct / total if total else 0.0
        print(f"== {label} ==", flush=True)
        print(f"Accuracy: {correct}/{total} ({accuracy:.2%})", flush=True)
        if summary["failures"]:
            print("First mismatches:", flush=True)
            for failure in summary["failures"][:5]:
                print(
                    _format_result_line(
                        failure["index"],
                        total,
                        failure["question"],
                        failure["expected"],
                        failure["got"],
                        False,
                    ),
                    flush=True,
                )
        else:
            print("All negative examples matched the expected response.", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
