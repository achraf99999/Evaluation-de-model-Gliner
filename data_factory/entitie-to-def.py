import json
from pathlib import Path

# --- Chemins vers les fichiers ---
train_path = Path("C:/Users/21650/Desktop/AI-Lawyer-main/train.json")
test_path = Path("C:/Users/21650/Desktop/AI-Lawyer-main/test.json")

# --- Dictionnaire des définitions ---
DEFINITIONS = {
    "DISO":    "Any deviation from the normal state of an organism: diseases, symptoms, dysfunctions, organ abnormalities (excluding injuries or poisoning).",
    "CHEM":    "Chemical substances, including legal/illegal drugs and biomolecules.",
    "DEVICE":  "Manufactured object used for medical or laboratory purposes.",
    "LABPROC": "Testing of body substances and other diagnostic procedures such as ultrasonography.",
    "PHYS":    "Biological function or process in an organism, including organism attributes (e.g., temperature), excluding mental processes.",
    "ANATOMY": "Organs, body parts, cells, cellular components and body substances.",
    "FINDING": "Statement conveying the results of a scientific observation or experiment.",
    "INJURY_POISONING":"damage inflicted on the body as the direct or indirect result"

}

def replace_codes_with_definitions(data):
    for example in data:
        new_ner = []
        for span in example.get("ner", []):
            start , code, end= span
            print(code)
            definition = DEFINITIONS.get(code, code)  # fallback = unchanged
            new_ner.append([start, end, definition])
            print(f"Remplacement de {code} par {definition} dans l'exemple.")
        example["ner"] = new_ner
    return data


# --- Traitement train ---
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_data = replace_codes_with_definitions(train_data)

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

# --- Traitement test ---
with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data = replace_codes_with_definitions(test_data)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print("✅ Codes remplacés par les définitions dans train.json et test.json.")
