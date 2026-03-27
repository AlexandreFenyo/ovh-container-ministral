import argparse
import os

import torch
from peft import AutoPeftModelForCausalLM
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "Ministral-8B-Instruct-2410"
DEFAULT_MODEL = "ministral-8b-instruct-lora"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful chatbot assistant for the Mon Espace Santé website. "
    "You answer only based on the information present in the FAQ."
)
DEFAULT_USER_PROMPT = (
    "Quelles sont les etapes pour le compte Mon espace sante de mon enfant "
    "qui vient d'atteindre la majorite ?"
)


def load_tokenizer(tokenizer_name):
    kwargs = {"use_fast": True}
    if int(transformers.__version__.split(".", 1)[0]) < 5:
        kwargs["fix_mistral_regex"] = True
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def load_model(model_name, model_kwargs):
    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        return AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_kwargs["device_map"],
            dtype=model_kwargs["dtype"],
        )
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Charge un modele Ministral fine-tune avec option de remplacement "
            "du nom du modele et du prompt utilisateur."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Nom du modele a charger via from_pretrained. "
            f"Si omis, utilise '{DEFAULT_MODEL}'."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=(
            "Nom du tokenizer a charger. "
            f"Par defaut: '{DEFAULT_BASE_MODEL}'."
        ),
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help=(
            "Contenu du message utilisateur a envoyer comme dernier message "
            "du contexte. Si omis, utilise le prompt par defaut."
        ),
    )
    args = parser.parse_args()

    model_name = args.model_name or DEFAULT_MODEL
    tokenizer = load_tokenizer(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto",
    )

    model = load_model(model_name, model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    user_prompt = args.user_prompt if args.user_prompt is not None else DEFAULT_USER_PROMPT
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    main()
