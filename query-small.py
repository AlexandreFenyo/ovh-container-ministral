import argparse
from contextlib import nullcontext
import os
import shlex
import sys

import torch
from peft import AutoPeftModelForCausalLM
from peft import PeftModel
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


def _build_training_style_messages(system_prompt, user_prompt):
    merged_user_prompt = (
        "Instructions:\n"
        f"{system_prompt.strip()}\n\n"
        "Question utilisateur:\n"
        f"{user_prompt.strip()}"
    )
    return [{"role": "user", "content": merged_user_prompt}]


def _discover_variants(adapter_root):
    variants = [("base", None)]
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


def _load_base_model(base_model_name, model_kwargs):
    return AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)


def _load_single_model(model_name, model_kwargs):
    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        return AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_kwargs["device_map"],
            dtype=model_kwargs["dtype"],
        )
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def _load_model_bundle(base_model_name, adapter_root, model_kwargs):
    variants = _discover_variants(adapter_root)
    base_model = _load_base_model(base_model_name, model_kwargs)
    adapter_variants = [(label, path) for label, path in variants if path is not None]
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


def _generate_with_model(model, tokenizer, system_prompt, user_prompt, max_new_tokens):
    messages = _build_training_style_messages(system_prompt, user_prompt)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def _generate_for_variant(
    model,
    tokenizer,
    system_prompt,
    user_prompt,
    max_new_tokens,
    variant_name,
    variants,
):
    variant_names = {label for label, _ in variants}
    if variant_name not in variant_names:
        raise ValueError(
            f"Unknown model variant {variant_name!r}. Available variants: {', '.join(label for label, _ in variants)}"
        )

    if variant_name == "base":
        context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with context:
            return _generate_with_model(
                model, tokenizer, system_prompt, user_prompt, max_new_tokens
            )

    if hasattr(model, "set_adapter"):
        model.set_adapter(variant_name)
    return _generate_with_model(
        model, tokenizer, system_prompt, user_prompt, max_new_tokens
    )


def _print_available_variants(variants):
    print("Available model variants:")
    for label, path in variants:
        if path is None:
            print(f"- {label}: base model without LoRA adapter")
        else:
            print(f"- {label}: {path}")


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
        help=(
            "Nom du modele a charger pour le mode simple. "
            "Si omis, utilise ft_output_dir du fichier de params."
        ),
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
    parser.add_argument(
        "--stdin-loop",
        action="store_true",
        help=(
            "Lit plusieurs questions sur l'entree standard, une par ligne, "
            "et repond a chacune independamment."
        ),
    )
    parser.add_argument(
        "--preload-variants",
        action="store_true",
        help=(
            "Charge une seule fois le modele de base et tous les adapters disponibles "
            "(base, checkpoints, final) pour permettre des comparaisons rapides."
        ),
    )
    parser.add_argument(
        "--query-all-variants",
        action="store_true",
        help=(
            "Pour chaque question, interroge automatiquement base, checkpoints et final, "
            "et affiche chaque reponse avec le nom du modele."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Nom de variante a utiliser avec --preload-variants en mode simple "
            "ou comme selection initiale avec --stdin-loop. Exemples: base, checkpoint-125, final."
        ),
    )
    args = parser.parse_args()

    if args.query_all_variants:
        args.preload_variants = True

    params_path = os.environ.get("PARAMS_CFG", "params-small.cfg")
    if not os.path.isabs(params_path):
        params_path = os.path.join(os.path.dirname(__file__), params_path)
    params = _load_params(params_path)

    base_model_name = _get_param(params, "model_name", required=False) or _get_param(
        params, "tokenizer"
    )
    adapter_root = _get_param(params, "ft_output_dir")
    tokenizer_name = args.tokenizer_name or _get_param(params, "tokenizer")
    system_prompt = args.system_prompt or _get_param(params, "system_prompt")
    user_prompt = args.user_prompt if args.user_prompt is not None else DEFAULT_USER_PROMPT

    tokenizer = _load_tokenizer(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        attn_implementation=_get_param(
            params, "attn_implementation", default="eager", required=False
        ),
        dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto",
    )

    if args.preload_variants:
        model, variants = _load_model_bundle(base_model_name, adapter_root, model_kwargs)
        active_variant = args.variant or ("final" if any(label == "final" for label, _ in variants) else "base")
        variant_names = {label for label, _ in variants}
        if active_variant not in variant_names:
            raise ValueError(
                f"Unknown variant {active_variant!r}. Available variants: {', '.join(label for label, _ in variants)}"
            )
    else:
        model_name = args.model_name or adapter_root
        model = _load_single_model(model_name, model_kwargs)
        variants = [("base", None)]
        active_variant = "base"

    model.config.pad_token_id = tokenizer.pad_token_id

    def answer_one(question):
        if args.query_all_variants:
            for label, _ in variants:
                response = _generate_for_variant(
                    model,
                    tokenizer,
                    system_prompt,
                    question,
                    args.max_new_tokens,
                    label,
                    variants,
                )
                print(f"[{label}]")
                print(response)
            return

        if args.preload_variants:
            response = _generate_for_variant(
                model,
                tokenizer,
                system_prompt,
                question,
                args.max_new_tokens,
                active_variant,
                variants,
            )
        else:
            response = _generate_with_model(
                model, tokenizer, system_prompt, question, args.max_new_tokens
            )
        print(response)

    if args.stdin_loop:
        if sys.stdin.isatty():
            print("Saisis une question par ligne. Ctrl-D pour quitter.")
            if args.preload_variants:
                print("Commandes disponibles: :models, :model <nom>")
                _print_available_variants(variants)
                print(f"Selected model variant: {active_variant}")

        for raw_line in sys.stdin:
            question = raw_line.strip()
            if not question:
                continue
            if args.preload_variants and question in {":models", ":list-models"}:
                _print_available_variants(variants)
                continue
            if args.preload_variants and question.startswith(":model "):
                selected = question.split(None, 1)[1].strip()
                variant_names = {label for label, _ in variants}
                if selected not in variant_names:
                    print(
                        f"Unknown variant: {selected}. Available variants: {', '.join(label for label, _ in variants)}"
                    )
                    continue
                active_variant = selected
                print(f"Selected model variant: {active_variant}")
                continue

            answer_one(question)
        return

    answer_one(user_prompt)


if __name__ == "__main__":
    main()
