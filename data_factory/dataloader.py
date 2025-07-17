#!/usr/bin/env python
# ── Build GLiNER-ready dataset (no tokenisation) ─────────────────────────────
import json, re, argparse
from pathlib import Path
from typing  import List, Dict

# ──────────────────────────────────────────────
_ROOT = Path(r"C:/Users/21650/Desktop/AI-Lawyer-main/"
             r"BioASQ_BIONNE_training_2024/DATASET_BIONNE/en")

_SUBSETS = ["train", "dev"]                       # dossiers à parcourir
_OUT_TMPL = "gliner_dataset_{split}.json"         # nom de sortie

ID_RE = re.compile(r"^(\d+)")                     # n° avant “_” ou “.”


# ══════════════════════════════════════════════
def extract_id(fname: str) -> str:
    """retourne l'identifiant numérique en tête du fichier"""
    m = ID_RE.match(fname)
    if not m:
        raise ValueError(f"Nom de fichier inattendu : {fname}")
    return m.group(1)


def parse_ann(ann_path: Path, text: str) -> List[Dict]:
    """lit .ann et retourne la liste des entités au format cible"""
    ents = []
    with ann_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.lstrip().startswith("T"):
                continue
            tid, infos, surface = line.rstrip().split("\t")
            label, *coords = infos.split()
            # plusieurs segments séparés par ';' = entité discontinue
            # → on les enregistre séparément
            for chunk in " ".join(coords).split(";"):
                start, end = map(int, chunk.split())
                ents.append({
                    "code_entity": label,
                    "spans": [start, end],
                    "entity": text[start:end]
                })
    return ents


def build_split(split_dir: Path, output_json: Path):
    txt_files = {extract_id(p.name): p
                 for p in split_dir.glob("*.txt")}
    ann_files = {extract_id(p.name): p
                 for p in split_dir.glob("*.ann")}

    examples = []
    for file_id, ann_path in ann_files.items():
        txt_path = txt_files.get(file_id)
        if not txt_path:
            print(f"⚠️  .ann sans .txt → ignoré : {ann_path.name}")
            continue

        text = txt_path.read_text(encoding="utf-8")
        entities = parse_ann(ann_path, text)

        examples.append({
            "text_id": file_id,
            "text": text,
            "entities": entities
        })

    output_json.write_text(json.dumps(examples,
                                      indent=2,
                                      ensure_ascii=False),
                           encoding="utf-8")
    print(f"✅ {len(examples)} exemples → {output_json}")


# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Convert .txt/.ann pairs into GLiNER JSON")
    parser.add_argument("--root", default=_ROOT, type=Path,
                        help="dossier racine contenant 'train' et 'dev'")
    args = parser.parse_args()

    for split in _SUBSETS:
        split_dir = args.root / split
        if not split_dir.exists():
            print(f"⚠️  Dossier absent : {split_dir}")
            continue
        build_split(split_dir,
                    split_dir / _OUT_TMPL.format(split=split))


if __name__ == "__main__":
    main()
