
import os
import shlex
import wandb
from huggingface_hub import login
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
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


def _get_param(params, key, alias=None, default=None, required=True):
    if key in params:
        return params[key]
    if alias and alias in params:
        return params[alias]
    if required:
        raise ValueError(f"Missing required parameter: {key}")
    return default


def _load_tokenizer(tokenizer_name):
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


def _fold_system_into_user(example):
    messages = example.get("messages", [])
    if len(messages) < 2:
        return example

    first_message = messages[0]
    second_message = messages[1]
    if first_message.get("role") != "system" or second_message.get("role") != "user":
        return example

    system_text = (first_message.get("content") or "").strip()
    user_text = (second_message.get("content") or "").strip()

    merged_user = dict(second_message)
    merged_user["content"] = (
        "Instructions:\n"
        f"{system_text}\n\n"
        "Question utilisateur:\n"
        f"{user_text}"
    )

    example["messages"] = [merged_user] + messages[2:]
    return example


params_path = os.environ.get("PARAMS_CFG", "params-small.cfg")
if not os.path.isabs(params_path):
    params_path = os.path.join(os.path.dirname(__file__), params_path)
params = _load_params(params_path)

known_keys = {
    "var_dataset_name",
    "var_wandb_project",
    "var_wandb_run",
    "wandb_notebook_name",
    "model_name",
    "tokenizer",
    "system_prompt",
    "attn_implementation",
    "ft_bf16",
    "ft_assistant_only_loss",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "lora_target_modules",
    "ft_learning_rate",
    "ft_gradient_checkpointing",
    "ft_num_train_epochs",
    "ft_logging_steps",
    "ft_per_device_train_batch_size",
    "ft_gradient_accumulation_steps",
    "ft_max_length",
    "ft_warmup_steps",
    "ft_lr_scheduler_type",
    "ft_output_dir",
    "ft_push_to_hub",
    "ft_report_to",
    "ft_eval_strategy",
    "ft_eval_steps",
}
unknown_keys = sorted(k for k in params.keys() if k not in known_keys)
if unknown_keys:
    print(f"Unknown keys in {params_path}: {', '.join(unknown_keys)}")

target_modules = _get_param(params, "lora_target_modules")
if isinstance(target_modules, str):
    target_modules = [target_modules]

resolved_params = {
    "var_dataset_name": _get_param(params, "var_dataset_name"),
    "var_wandb_project": _get_param(params, "var_wandb_project"),
    "var_wandb_run": _get_param(params, "var_wandb_run"),
    "wandb_notebook_name": _get_param(params, "wandb_notebook_name"),
    "model_name": _get_param(params, "model_name", default=None, required=False),
    "tokenizer": _get_param(params, "tokenizer"),
    "attn_implementation": _get_param(params, "attn_implementation", default="eager", required=False),
    "ft_bf16": _get_param(params, "ft_bf16", default=True, required=False),
    "ft_assistant_only_loss": _get_param(params, "ft_assistant_only_loss", default=True, required=False),
    "lora_r": _get_param(params, "lora_r"),
    "lora_alpha": _get_param(params, "lora_alpha"),
    "lora_dropout": _get_param(params, "lora_dropout"),
    "lora_bias": _get_param(params, "lora_bias"),
    "lora_target_modules": target_modules,
    "ft_learning_rate": _get_param(params, "ft_learning_rate"),
    "ft_gradient_checkpointing": _get_param(params, "ft_gradient_checkpointing"),
    "ft_num_train_epochs": _get_param(params, "ft_num_train_epochs"),
    "ft_logging_steps": _get_param(params, "ft_logging_steps"),
    "ft_per_device_train_batch_size": _get_param(params, "ft_per_device_train_batch_size"),
    "ft_gradient_accumulation_steps": _get_param(params, "ft_gradient_accumulation_steps"),
    "ft_max_length": _get_param(params, "ft_max_length"),
    "ft_warmup_steps": _get_param(params, "ft_warmup_steps"),
    "ft_lr_scheduler_type": _get_param(params, "ft_lr_scheduler_type"),
    "ft_output_dir": _get_param(params, "ft_output_dir"),
    "ft_push_to_hub": _get_param(params, "ft_push_to_hub"),
    "ft_report_to": _get_param(params, "ft_report_to"),
    "ft_eval_strategy": _get_param(params, "ft_eval_strategy"),
    "ft_eval_steps": _get_param(params, "ft_eval_steps"),
}

print(f"Config values from {params_path}:")
for key in sorted(resolved_params.keys()):
    print(f"  {key}={resolved_params[key]}")

var_dataset_name = resolved_params["var_dataset_name"]
var_wandb_project = resolved_params["var_wandb_project"]
var_wandb_run = resolved_params["var_wandb_run"]
wandb_notebook_name = resolved_params["wandb_notebook_name"]
tokenizer_name = resolved_params["tokenizer"]
model_name = resolved_params["model_name"] or tokenizer_name

model_kwargs = dict(
    attn_implementation=resolved_params["attn_implementation"],
    dtype=torch.bfloat16,
    use_cache=False,
)

os.environ['WANDB_NOTEBOOK_NAME'] = wandb_notebook_name
wandb.login(key=os.environ['wandbkey'])
login(token=os.environ['hfkey'])
wandb.init(project=var_wandb_project, entity="alexandre-fenyo-fenyonet", name=var_wandb_run)

# Chargement du tokenizer
tokenizer = _load_tokenizer(tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Chargement du modèle
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
model.config.pad_token_id = tokenizer.pad_token_id

# Chargement des jeux de données
train_dataset = load_dataset(var_dataset_name, split="train")
# Conserve uniquement la colonne "messages" (ou adapte selon ton schéma)
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "messages"])
train_dataset = train_dataset.map(_fold_system_into_user)

eval_dataset = load_dataset(var_dataset_name, split="validation")
eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c != "messages"])
eval_dataset = eval_dataset.map(_fold_system_into_user)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=resolved_params["lora_r"],
    lora_alpha=resolved_params["lora_alpha"],
    lora_dropout=resolved_params["lora_dropout"],
    bias=resolved_params["lora_bias"],
    target_modules=target_modules,
)

training_args = SFTConfig(
    learning_rate=resolved_params["ft_learning_rate"],
    bf16=resolved_params["ft_bf16"],
    gradient_checkpointing=resolved_params["ft_gradient_checkpointing"],
    num_train_epochs=resolved_params["ft_num_train_epochs"],
    logging_steps=resolved_params["ft_logging_steps"],
    per_device_train_batch_size=resolved_params["ft_per_device_train_batch_size"],
    gradient_accumulation_steps=resolved_params["ft_gradient_accumulation_steps"],
    max_length=resolved_params["ft_max_length"],
    warmup_steps=resolved_params["ft_warmup_steps"],
    lr_scheduler_type=resolved_params["ft_lr_scheduler_type"],
    output_dir=resolved_params["ft_output_dir"],
    push_to_hub=resolved_params["ft_push_to_hub"],
    report_to=resolved_params["ft_report_to"],
    eval_strategy=resolved_params["ft_eval_strategy"],
    eval_steps=resolved_params["ft_eval_steps"],
    assistant_only_loss=resolved_params["ft_assistant_only_loss"],
    save_strategy="epoch",
    save_total_limit=100,
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)
# trainer.push_to_hub(dataset_name=var_dataset_name)
