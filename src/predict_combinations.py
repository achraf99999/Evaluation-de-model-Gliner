import json
import itertools
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch

from gliner import GLiNER
from src.config import ENTITY_TYPES, DATA_PATH, MODEL_NAME, THRESHOLD, PRED_COMBINATIONS_JSON, MIN_COMB, MAX_COMB
from src.utils import load_json

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Loading GLiNER on {device}")
    return GLiNER.from_pretrained(MODEL_NAME, device=device)

def main():
    model = load_model()
    dataset = load_json(DATA_PATH)

    debug_combinations = {}

    for entity_code, synonyms in ENTITY_TYPES.items():
        max_comb = MAX_COMB or len(synonyms)
        combinations = []
        for k in range(MIN_COMB, max_comb + 1):
            combinations.extend(itertools.combinations(synonyms, k))

        for combo in combinations:
            combo_list = list(combo)
            combo_key = "__".join(combo_list)
            full_key = f"{entity_code}__{combo_key}"

            print(f" {full_key}")
            debug_entries = []

            for doc in tqdm(dataset, desc=combo_key, unit="doc"):
                text = doc["text"]
                preds = model.predict_entities(text, combo_list, threshold=THRESHOLD)

                for ent in preds:
                    if ent["label"] in combo_list:
                        debug_entries.append({
                            "text_id": doc.get("text_id", ""),
                            "text": text,
                            "span": [ent["start"], ent["end"]],
                            "entity_text": text[ent["start"]:ent["end"]],
                            "label": ent["label"]
                        })

            debug_combinations[full_key] = debug_entries

    PRED_COMBINATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_COMBINATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(debug_combinations, f, indent=2, ensure_ascii=False)

    print(f"\n Saved combination-level predictions to: {PRED_COMBINATIONS_JSON}")

if __name__ == "__main__":
    main()
