import argparse
import os
import shutil

import torch
from huggingface_hub import hf_hub_download, save_torch_model
from peft import PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_ID = "Ministral-8B-Instruct-2410"
LORA_ADAPTER_ID = "ministral-8b-instruct-lora"


def maybe_copy_file_from_hub(repo_id: str, filename: str, out_dir: str) -> None:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(out_dir, filename))


def load_tokenizer_from_pretrained(repo_id: str) -> AutoTokenizer:
    kwargs = {"use_fast": True}
    if int(transformers.__version__.split(".", 1)[0]) < 5:
        kwargs["fix_mistral_regex"] = True
    try:
        return AutoTokenizer.from_pretrained(repo_id, **kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(repo_id, **kwargs)


def load_tokenizer() -> AutoTokenizer:
    for repo_id in (LORA_ADAPTER_ID, BASE_MODEL_ID):
        try:
            return load_tokenizer_from_pretrained(repo_id)
        except Exception:
            pass
    raise RuntimeError("Impossible de charger le tokenizer depuis le modele base ou l'adaptateur LoRA.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./ministral-8b-instruct-merged")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_shard_size", type=str, default="2GB")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    tokenizer = load_tokenizer()

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
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

    maybe_copy_file_from_hub(LORA_ADAPTER_ID, "chat_template.jinja", args.out_dir)

    print(f"OK: modele merge sauvegarde dans: {args.out_dir}")


if __name__ == "__main__":
    main()
