# ovh-container-ministral

Pipeline minimal de fine-tuning LoRA pour `Ministral-8B-Instruct-2410`, centré sur une FAQ Mon Espace Santé.

Le dépôt contient trois scripts principaux :

- `ft-small.py` : entraîne un adaptateur LoRA avec TRL `SFTTrainer`
- `merge-small.py` : fusionne le modèle base et l'adaptateur LoRA
- `query-small.py` : teste l'inférence sur l'adaptateur ou sur le modèle fusionné

La configuration d'entraînement est dans `params-small.cfg`.

## Pré-requis

- Python 3
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
├── ft-small.py
├── merge-small.py
├── query-small.py
├── params-small.cfg
└── Makefile
```

## Notes

- `Makefile` contient encore un workflow Docker/OVH partiel, distinct des scripts Python présents dans ce dépôt.
- Le champ `system_prompt` de `params-small.cfg` est informatif ; le comportement appris dépend du contenu réel du dataset.
- Les répertoires de sortie `ministral-8b-instruct-lora/` et `ministral-8b-instruct-merged/` sont ignorés par git.
