# enrich_kb_with_llm.py

import json
from pathlib import Path
import google.generativeai as genai
import os

def enrich_knowledge_base_with_llm(input_kb_path: Path, output_kb_path: Path):
    """
    Charge une base de connaissances existante, génère des requêtes pour un LLM
    afin d'obtenir des définitions et des concepts liés, et sauvegarde la KB enrichie.

    ATTENTION : Ce script nécessite une clé API Gemini configurée pour fonctionner.
    """
    print(f"Chargement de la base de connaissances depuis : {input_kb_path}")
    try:
        with open(input_kb_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{input_kb_path}' n'a pas été trouvé. Veuillez vous assurer qu'il existe.")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier '{input_kb_path}' n'est pas un JSON valide.")
        return

    # --- Configuration du LLM ---

    #genai.configure(api_key="Votre clé API ici")   (à des fins de test uniquement)

    # Initialisez votre modèle LLM
    llm_model = genai.GenerativeModel('gemini-2.0-flash')
    # -------------------------------------------------------------

    print("Début de l'enrichissement de la base de connaissances avec le LLM...")

    for i, concept in enumerate(knowledge_base):
        concept_id = concept.get("kb_id", f"Concept_{i}")
        label = concept.get("label", "Concept Inconnu")
        concept_type = concept.get("type", "Non Spécifié")
        synonyms = ", ".join(concept.get("synonyms", []))

        # Construction du prompt pour le LLM
        prompt_text = (
            f"Fournis une brève définition encyclopédique pour le concept médical/scientifique '{label}' "
            f"(type : {concept_type}, synonymes : {synonyms}). "
            "Inclue également 3 à 5 concepts étroitement liés à celui-ci, avec leur 'label' et 'type' si possible. "
            "La réponse doit être au format JSON avec les clés 'definition' (string) et 'related_concepts' (liste de dictionnaires). "
            "Chaque dictionnaire dans 'related_concepts' doit avoir 'label' et 'type' (ex: {\"label\": \"Épilepsie\", \"type\": \"DISO\"}).\n"
            "Assure-toi que la réponse est un JSON valide et propre, sans aucun texte supplémentaire avant ou après le bloc JSON.\n"
            "Exemple de format JSON attendu:\n"
            "{\n"
            "  \"definition\": \"Une brève explication du concept ici.\",\n"
            "  \"related_concepts\": [\n"
            "    {\"label\": \"Concept lié 1\", \"type\": \"TYPE_DU_CONCEP\"},\n"
            "    {\"label\": \"Concept lié 2\", \"type\": \"TYPE_DU_CONCEP\"}\n"
            "  ]\n"
            "}"
        )

        print(f"\n--- Requête pour {concept_id} ({label}) ---")
        # print(f"Prompt généré:\n{prompt_text}\n") # Décommenter pour voir le prompt complet

        try:
            response = llm_model.generate_content(prompt_text)
            
            # Nettoyage de la réponse pour extraire le JSON pur
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")].strip()
            
            generated_data = json.loads(response_text)
            
            # Assigner les valeurs générées aux champs du concept
            concept["definition"] = generated_data.get("definition", "")
            concept["related_concepts"] = generated_data.get("related_concepts", [])

            print(f"  Concept '{label}' enrichi avec succès.")

        except json.JSONDecodeError as e:
            print(f"  Erreur de parsing JSON pour '{label}' : {e}. Réponse brute:\n{response_text[:500]}...")
            concept["definition"] = "Erreur de format JSON lors de la génération."
            concept["related_concepts"] = []
        except Exception as e:
            print(f"  Erreur lors de l'appel LLM ou traitement pour '{label}' : {e}")
            concept["definition"] = "Erreur lors de l'enrichissement par LLM."
            concept["related_concepts"] = []

    print(f"\nSauvegarde de la base de connaissances enrichie vers : {output_kb_path}")
    with open(output_kb_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print("Processus d'enrichissement terminé.")


if __name__ == "__main__":
    # Définition des chemins des fichiers (à ajuster si nécessaire)
    INPUT_KB_FILE = Path("knowledge_base.json") # Le fichier généré par le premier script
    OUTPUT_KB_FILE = Path("knowledge_base_enriched.json") # Le fichier de sortie enrichi

    enrich_knowledge_base_with_llm(INPUT_KB_FILE, OUTPUT_KB_FILE)