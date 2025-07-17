from pathlib import Path

# ════════════════════ ENTITÉS ET SYNONYMES ════════════════════
ENTITY_TYPES = {
    "DISO":  ["disease", "disorder", "syndrome", "pathology"],
    "CHEM":  ["chemical", "compound", "substance", "medication", "drug"],
    "DEVICE": ["device", "apparatus", "equipment"],
    "LABPROC": ["procedure", "test", "examination"],
    "PHYS":  ["physiology", "biological process", "bodily function"],
    "ANATOMY": ["anatomy", "body part", "organ"],
    "FINDING": ["finding", "observation", "result"],
    "INJURY_POISONING": ["injury", "poisoning", "tension of ligaments"],
}

# ════════════════════ CHEMINS PRINCIPAUX ════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "fulldata.json"

OUTPUT_DIR = BASE_DIR / "outputs"
PRED_SYNONYM_JSON = OUTPUT_DIR / "debug_by_synonym.json"
PRED_COMBINATIONS_JSON = OUTPUT_DIR / "debug_combinations.json"

UNION_DIR = OUTPUT_DIR / "results_union"
INTERSECTION_DIR = OUTPUT_DIR / "results_intersection"
OVERLAP_DIR = OUTPUT_DIR / "overlap_analysis"

# ════════════════════ MODÈLE ET SEUILS ════════════════════
MODEL_NAME = "knowledgator/gliner-bi-small-v1.0"
THRESHOLD = 0.5   # seuil de prédiction
DEFAULT_JACCARD_THRESHOLD = 0.5  # seuil pour l'évaluation partielle

# ════════════════════ COMBINATOIRE ════════════════════
MIN_COMB = 1
MAX_COMB = None  
