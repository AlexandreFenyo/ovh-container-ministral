# ovh-container-ministral

Pipeline minimal de fine-tuning LoRA pour `Ministral-8B-Instruct-2410`, centré sur une FAQ Mon Espace Santé.

Le dépôt contient trois scripts principaux :

- `ft-small.py` : entraîne un adaptateur LoRA avec TRL `SFTTrainer`
- `merge-small.py` : fusionne le modèle base et l'adaptateur LoRA
- `query-small.py` : teste l'inférence sur l'adaptateur ou sur le modèle fusionné

La configuration d'entraînement est dans `params-small.cfg`.

## Pré-requis

- Python 3.13.12 recommandé
- accès GPU recommandé pour l'entraînement
- un compte Hugging Face avec accès au modèle `Ministral-8B-Instruct-2410`
- un compte Weights & Biases si tu gardes le reporting W&B activé

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

## Structure

```text
.
├── .python-version
├── ft-small.py
├── merge-small.py
├── params-small.cfg
├── query-small.py
├── requirements.txt
└── Makefile
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
% python convert_hf_to_gguf.py ../ovh-container-ministral/ministral-8b-instruct-merged --outfile ../ovh-container-ministral/ministral-8b-instruct-merged.gguf --outtype f16
INFO:hf-to-gguf:Model successfully exported to ../ovh-container-ministral/ministral-8b-instruct-merged.gguf
% cat Modelfile.ollama
FROM ministral-8b-instruct-merged.gguf
% ollama create ministral-merged -f Modelfile.ollama
```

## Notes

- `Makefile` contient encore un workflow Docker/OVH partiel, distinct des scripts Python présents dans ce dépôt.
- Le champ `system_prompt` de `params-small.cfg` est informatif ; le comportement appris dépend du contenu réel du dataset.
- Les répertoires de sortie `ministral-8b-instruct-lora/` et `ministral-8b-instruct-merged/` sont ignorés par git.
