# ovh-container-ministral

Pipeline minimal de fine-tuning LoRA pour `Ministral-8B-Instruct-2410`, centré sur une FAQ Mon Espace Santé.

Le dépôt contient quatre scripts principaux :

- `ft-small.py` : entraîne un adaptateur LoRA avec TRL `SFTTrainer`
- `merge-small.py` : fusionne le modèle base et l'adaptateur LoRA
- `query-small.py` : teste l'inférence sur l'adaptateur ou sur le modèle fusionné
- `validate-small.py` : vérifie les réponses attendues sur les exemples `negative`
- `validate-positive-small.py` : juge les réponses sur les exemples `positive` avec OVH

La configuration d'entraînement est dans `params-small.cfg`.

## Pré-requis

- Python 3.13.12 recommandé
- accès GPU recommandé pour l'entraînement
- un compte Hugging Face avec accès au modèle `Ministral-8B-Instruct-2410`
- un compte Weights & Biases si tu gardes le reporting W&B activé

Variables d'environnement avec les jetons d'API : `source /mnt/e/AI_experiments/env.sh`

Bibliothèques Python attendues par les scripts :

- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`
- `wandb`
- `huggingface_hub`

Le dépôt inclut un `requirements.txt` avec un jeu de versions récent et conservateur pour le ML, prévu pour Python 3.13.

## Environnement local

Convention recommandée pour ce dépôt : environnement virtuel dans `.venv/`.

Création et activation :

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Le répertoire `.venv/` est ignoré par git.

## Variables d'environnement

Les scripts d'entraînement utilisent :

- `hfkey` : token Hugging Face
- `wandbkey` : token Weights & Biases
- `PARAMS_CFG` : chemin optionnel vers un autre fichier de configuration
- `OVH_AI_TOKEN` : token pour le juge OVH utilisé par `validate-positive-small.py`

Exemple :

```bash
export hfkey=...
export wandbkey=...
```

## Entraînement

Le script charge le dataset Hugging Face défini dans `params-small.cfg`, prépare le modèle base, applique le LoRA puis entraîne l'adaptateur.

```bash
python3 ft-small.py
```

Pour utiliser un autre fichier de paramètres :

```bash
PARAMS_CFG=mon-fichier.cfg python3 ft-small.py
```

## Fusion des poids

Après entraînement, fusion du modèle base et de l'adaptateur :

```bash
python3 merge-small.py --out_dir ./ministral-8b-instruct-merged
```

## Test d'inférence

Test sur l'adaptateur LoRA local :

```bash
python3 query-small.py --model-name ministral-8b-instruct-lora
```

Test sur le modèle fusionné :

```bash
python3 query-small.py --model-name ./ministral-8b-instruct-merged
```

Avec un prompt personnalisé :

```bash
python3 query-small.py \
  --model-name ./ministral-8b-instruct-merged \
  --user-prompt "Comment activer Mon espace sante ?"
```

## Validation du finetuning

Le validateur charge le dataset, filtre les exemples `negative`, puis compare les réponses générées par chaque checkpoint et par le modèle final à la réponse attendue.

Avec un split explicite :

```bash
python3 validate-small.py --split validation
```

Pour tester rapidement sur seulement deux lignes négatives :

```bash
python3 validate-small.py --split validation --max-negative-examples 2
```

Sans `--split`, les splits disponibles sont affichés puis le script demande lequel analyser.

## Validation positive avec juge OVH

Le validateur positif charge les exemples `positive`, génère une réponse pour chaque checkpoint et le modèle final, puis demande à un juge OVH `gpt-oss-120b` de scorer la réponse.

Avec un split explicite :

```bash
python3 validate-positive-small.py --split validation
```

Pour tester rapidement sur seulement deux lignes positives :

```bash
python3 validate-positive-small.py --split validation --max-positive-examples 2
```

Sans `--split`, les splits disponibles sont affichés puis le script demande lequel analyser.

## Structure

```text
.
├── .python-version
├── ft-small.py
├── merge-small.py
├── validate-positive-small.py
├── validate-small.py
├── params-small.cfg
├── query-small.py
└── requirements.txt
```

## Ajouter le message système en dur dans le chat template, après le merge
Modifier `ministral-8b-instruct-merged/chat_template.jinja` en supprimant :
```jinja2
{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
```
Et en le remplaçant par :
```jinja2
{%- set system_message = "You are a helpful chatbot assistant for the Mon Espace Santé website. You answer only based on the information present in the FAQ. If the information is not available, you must respond with the predefined refusal message and nothing else." %}
{%- set loop_messages = messages %}
```


## Transformer en GGUF pour Ollama et l'importer dans Ollama

Utiliser un autre répertoire et un autre env, en clonant llama.cpp et en mettant à jour le tokenizer du modèle comme ceci :
```bash
% cd git
% cd llama.cpp
% source .venv/bin/activate
% cd ../ovh-container-ministral/ministral-8b-instruct-merged/
% sed -i 's/"tokenizer_class": "TokenizersBackend"/"tokenizer_class": "PreTrainedTokenizerFast"/' tokenizer_config.json
% cd -
% rm -rf ../ovh-container-ministral/ministral-8b-instruct-merged.gguf
% python convert_hf_to_gguf.py ../ovh-container-ministral/ministral-8b-instruct-merged --outfile ../ovh-container-ministral/ministral-8b-instruct-merged.gguf --outtype f16
INFO:hf-to-gguf:Model successfully exported to ../ovh-container-ministral/ministral-8b-instruct-merged.gguf
NOTE : PASSER SOUS cygwin ici
% cat Modelfile.ollama
FROM ministral-8b-instruct-merged.gguf
% ollama rm ministral-merged
% ollama create ministral-merged -f Modelfile.ollama
```

## Notes

- Le champ `system_prompt` de `params-small.cfg` est informatif ; le comportement appris dépend du contenu réel du dataset.
- Les répertoires de sortie `ministral-8b-instruct-lora/` et `ministral-8b-instruct-merged/` sont ignorés par git.
