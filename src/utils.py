import json
import torch
from pathlib import Path
from gliner import GLiNER

# Chargement du corpus depuis un fichier JSON
def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Chargement du modèle GLiNER depuis HuggingFace ou local
def load_gliner(model_name: str = "knowledgator/gliner-bi-small-v1.0"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Chargement de GLiNER : {model_name} sur {device}")
    return GLiNER.from_pretrained(model_name, device=device)

# Calcul de l'indice de Jaccard entre deux ensembles
def jaccard(set_a: set, set_b: set) -> float:
    return len(set_a & set_b) / len(set_a | set_b) if (set_a or set_b) else 0.0

# Calcul précision, rappel, F1 à partir de TP, FP, FN
def prf1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1

# Sauvegarde JSON
def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Chargement JSON
import json
from pathlib import Path
from typing import List, Any

def load_json(file_path: Path) -> Any:
    """Charge un fichier JSON sans restrictions de type"""
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def ensure_dir(path: Path):
    """
    Assure que le répertoire spécifié existe. Si ce n'est pas le cas, il est créé.
    """
    path.mkdir(parents=True, exist_ok=True)