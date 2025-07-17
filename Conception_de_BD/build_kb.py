import json
from pathlib import Path
from collections import defaultdict

# --- Réplication des parties essentielles de src/config.py ---
# ATTENTION : Ajustez ce chemin si votre fichier data.json n'est pas directement
# dans le même répertoire que ce script, ou dans un sous-dossier 'data'.
# Par exemple, si votre fichier s'appelle 'fulldata.json' et est dans 'data/fulldata.json':
# DATA_PATH = Path("data/fulldata.json")
# Pour l'exemple basé sur votre question, nous utiliserons "data.json"
DATA_PATH = Path("C:\\Users\\21650\\Desktop\\pfe\\data\\fulldata.json")

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

def build_knowledge_base(data_path: Path, entity_types_config: dict) -> list:
    """
    Construit une base de connaissances à partir d'un fichier data.json annoté.
    Cette fonction réalise une première étape de normalisation simple
    en regroupant les mentions textuelles identiques pour un même code d'entité.

    Note : Une curation manuelle sera nécessaire pour regrouper les synonymes
    (ex: "seizure" et "epileptic seizure" devraient être un seul concept "Crise d'épilepsie")
    et pour ajouter les définitions et les relations.
    """
    print(f"Chargement des données depuis : {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{data_path}' n'a pas été trouvé. Veuillez vérifier le chemin.")
        return []
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier '{data_path}' n'est pas un JSON valide.")
        return []

    # Structure temporaire pour stocker les concepts uniques avant la finalisation.
    # Clé: (forme_normalisée_du_libellé, type_entité) -> {données du concept}
    concepts_staging = {}
    kb_concepts_list = []
    # Compteur pour générer des IDs uniques par type d'entité (ex: DISO_000, DISO_001)
    current_kb_id_counter = defaultdict(int)

    print("Extraction des mentions et regroupement initial par code d'entité et texte...")

    for doc in data:
        # Assurez-vous que la clé 'entities' existe dans chaque document
        if "entities" in doc and isinstance(doc["entities"], list):
            for entity_entry in doc["entities"]:
                entity_text = entity_entry.get("entity", "").strip()
                code_entity = entity_entry.get("code_entity", "").strip()

                # Ignorer les entrées incomplètes ou vides
                if not entity_text or not code_entity:
                    continue

                # Utilisation d'une version en minuscules pour la clé de regroupement
                # Cela permet de traiter "Seizure" et "seizure" comme la même mention textuelle.
                # Pour une normalisation sémantique plus poussée (ex: "seizure" et "epileptic seizure" -> "Crise d'épilepsie"),
                # cette logique devrait être améliorée, potentiellement avec un dictionnaire de mapping manuel.
                normalized_text_for_key = entity_text.lower()

                # Clé unique pour identifier ce concept spécifique dans notre zone de préparation
                concept_key = (normalized_text_for_key, code_entity)

                if concept_key not in concepts_staging:
                    # Si c'est la première fois que nous rencontrons cette combinaison texte/code,
                    # nous créons un nouveau concept dans notre KB en préparation.
                    kb_id = f"{code_entity}_{str(current_kb_id_counter[code_entity]).zfill(3)}"
                    concepts_staging[concept_key] = {
                        "kb_id": kb_id,
                        "label": entity_text, # Le premier texte rencontré devient le label initial
                        "type": code_entity,
                        "synonyms": set(), # Utilisation d'un set pour éviter les doublons de synonymes
                        "definition": "", # À remplir manuellement plus tard
                        "related_concepts": [] # À remplir manuellement plus tard
                    }
                    current_kb_id_counter[code_entity] += 1 # Incrémenter le compteur pour ce type d'entité

                # Ajout de la mention textuelle actuelle comme synonyme au concept correspondant.
                concepts_staging[concept_key]["synonyms"].add(entity_text)

    print("Intégration des synonymes définis dans config.py...")

    # Parcours des types d'entités et de leurs synonymes définis dans le fichier config.py.
    # L'objectif est d'ajouter ces synonymes à la KB, soit en les liant à des concepts existants,
    # soit en créant de nouveaux concepts s'ils n'ont pas été détectés dans data.json.
    for entity_type, synonyms_list_from_config in entity_types_config.items():
        for syn_from_config in synonyms_list_from_config:
            normalized_syn_for_key = syn_from_config.lower().strip()
            concept_key = (normalized_syn_for_key, entity_type)

            if concept_key not in concepts_staging:
                # Si un synonyme de config.py n'a pas été trouvé comme mention dans data.json
                # (ou s'il est une nouvelle forme canonique), on le crée comme un nouveau concept.
                kb_id = f"{entity_type}_{str(current_kb_id_counter[entity_type]).zfill(3)}"
                concepts_staging[concept_key] = {
                    "kb_id": kb_id,
                    "label": syn_from_config, # Le synonyme de config.py devient le label canonique ici
                    "type": entity_type,
                    "synonyms": {syn_from_config}, # Commence avec lui-même comme synonyme
                    "definition": "",
                    "related_concepts": []
                }
                current_kb_id_counter[entity_type] += 1
            else:
                # Si le concept existe déjà (basé sur une mention de data.json),
                # on ajoute simplement le synonyme de config.py à sa liste.
                concepts_staging[concept_key]["synonyms"].add(syn_from_config)

    # Finalisation : conversion des sets de synonymes en listes triées et création de la liste finale de concepts.
    for concept_data in concepts_staging.values():
        concept_data["synonyms"] = sorted(list(concept_data["synonyms"]))
        kb_concepts_list.append(concept_data)

    # Tri de la liste finale des concepts par kb_id pour une meilleure organisation.
    kb_concepts_list.sort(key=lambda x: x["kb_id"])

    print(f"Base de connaissances initiale construite avec {len(kb_concepts_list)} concepts.")
    return kb_concepts_list

# Point d'entrée principal du script
if __name__ == "__main__":
    # Exécute la fonction pour construire la base de connaissances.
    kb = build_knowledge_base(DATA_PATH, ENTITY_TYPES)

    # Sauvegarde la base de connaissances résultante dans un fichier JSON.
    kb_output_path = Path("knowledge_base.json")
    with open(kb_output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False) # indent pour une meilleure lisibilité, ensure_ascii=False pour caractères UTF-8

    print(f"Base de connaissances sauvegardée dans : {kb_output_path}")