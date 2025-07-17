import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch

from gliner import GLiNER
from src.config import ENTITY_TYPES, DATA_PATH, MODEL_NAME, THRESHOLD, PRED_SYNONYM_JSON
from src.utils import load_json

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Loading GLiNER on {device}")
    return GLiNER.from_pretrained(MODEL_NAME, device=device)

def main():
    model = load_model()
    dataset = load_json(DATA_PATH)

    debug_data = defaultdict(list)

    for entity_code, synonyms in ENTITY_TYPES.items():
        for synonym in synonyms:
            print(f" Processing {entity_code} – {synonym}")
            for doc in tqdm(dataset, desc=f"{entity_code} – {synonym}", unit="doc"):
                text = doc["text"]
                preds = model.predict_entities(text, [synonym], threshold=THRESHOLD)

                for ent in preds:
                    if ent["label"] == synonym:
                        debug_data[f"{entity_code}__{synonym}"].append({
                            "text_id": doc.get("text_id", ""),
                            "text": text,
                            "span": [ent["start"], ent["end"]],
                            "entity_text": text[ent["start"]:ent["end"]],
                            "label": synonym
                        })

    PRED_SYNONYM_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_SYNONYM_JSON, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)

    print(f"\n Saved synonym-level predictions to: {PRED_SYNONYM_JSON}")

if __name__ == "__main__":
    main()
