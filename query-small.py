import argparse
import os
import shlex

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_USER_PROMPT = (
    "Quelles sont les etapes pour le compte Mon espace sante de mon enfant "
    "qui vient d'atteindre la majorite ?"
)


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


def _load_model(model_name, model_kwargs):
    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        return AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_kwargs["device_map"],
            dtype=model_kwargs["dtype"],
        )
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def _build_training_style_messages(system_prompt, user_prompt):
    merged_user_prompt = (
        "Instructions:\n"
        f"{system_prompt.strip()}\n\n"
        "Question utilisateur:\n"
        f"{user_prompt.strip()}"
    )
    return [{"role": "user", "content": merged_user_prompt}]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Charge un modele Ministral fine-tune avec la meme mise en forme "
            "de prompt que ft-small.py."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nom du modele a charger. Si omis, utilise ft_output_dir du fichier de params.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Nom du tokenizer a charger. Si omis, utilise tokenizer du fichier de params.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Prompt systeme a utiliser. Si omis, utilise system_prompt du fichier de params.",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help="Question utilisateur a envoyer. Si omis, utilise le prompt par defaut.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Nombre maximum de tokens a generer.",
    )
    args = parser.parse_args()

    params_path = os.environ.get("PARAMS_CFG", "params-small.cfg")
    if not os.path.isabs(params_path):
        params_path = os.path.join(os.path.dirname(__file__), params_path)
    params = _load_params(params_path)

    model_name = args.model_name or _get_param(params, "ft_output_dir")
    tokenizer_name = args.tokenizer_name or _get_param(params, "tokenizer")
    system_prompt = args.system_prompt or _get_param(params, "system_prompt")
    user_prompt = args.user_prompt if args.user_prompt is not None else DEFAULT_USER_PROMPT

    tokenizer = _load_tokenizer(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        attn_implementation=_get_param(params, "attn_implementation", default="eager", required=False),
        dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto",
    )

    model = _load_model(model_name, model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    messages = _build_training_style_messages(system_prompt, user_prompt)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    main()
