from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pandas as pd # Ajout de l'importation de pandas pour la gestion des DataFrames

# --- 1. Définir les phrases et mots cibles pour l'expérience ---
test_cases = [

    {
        "sentence": "Today ,i get the result of my blood tests that revealed the presence of a specific chemical biomarker.",
        "target_word": "chemical"
    },
    {
        "sentence": "Biomedical devices are   ubiquitous in modern life and play a vital role in the restoration of biological functionality. ",
        "target_word": "devices"
    },
    {
        "sentence": "A further subheading entitled procedure or interventions may  also be considered for clinical trials.",
        "target_word": "procedure"
    },
    {
        "sentence": "Physiology is a core  element of an undergraduate biomedical engineering curriculum",
        "target_word": "Physiology"
    },
    {
        "sentence": "There is a perceived need for anatomy instruction for graduate students enrolled in a biomedical engineering program. ",
        "target_word": "anatomy"
    },
    {
        "sentence": "This finding should be taken into account in the choice of the   optimal surgical approach to the maxillary sinus.",
        "target_word": "finding"
    },
    {
        "sentence": "Lung sound may be used to diagnose  lung injury in pregnant women with blood cancers.",
        "target_word": "injury"
    },
    {
        "sentence": "any deviations from   normal state of organism: diseases, symptoms, abnormality of appendicitis, haemorrhoids, magnesium deficiency dysfuncorgan, excluding injuries or poisoning",
        "target_word": "diseases"
    }
]


# --- 2. Configuration du Modèle de Langage Masqué (MLM) ---

model_name_mlm = "bert-base-uncased"
tokenizer_mlm = AutoTokenizer.from_pretrained(model_name_mlm)
model_mlm = AutoModelForMaskedLM.from_pretrained(model_name_mlm)

# Création d'un pipeline pour faciliter l'utilisation du modèle de remplissage de masque
unmasker = pipeline('fill-mask', model=model_mlm, tokenizer=tokenizer_mlm)

# --- 3. Configuration du Modèle Sentence Transformer pour la similarité sémantique ---

model_name_st = "all-MiniLM-L6-v2"
model_st = SentenceTransformer(model_name_st)


# --- 4. Fonction principale pour exécuter l'analyse pour un cas donné ---
def analyze_word_in_sentence(sentence, target_word, unmasker, model_st):

    # Masquer le mot cible dans la phrase
    # Remplace la première occurrence du mot cible par le token de masque du tokenizer
    masked_sentence = sentence.replace(target_word, unmasker.tokenizer.mask_token, 1)

    print(f"\n--- Traitement du mot '{target_word}' dans la phrase ---")
    print(f"Phrase originale : {sentence}")
    print(f"Phrase masquée : {masked_sentence}")

    # Obtenir les prédictions du MLM pour le mot masqué
    predictions = unmasker(masked_sentence, top_k=10)

    # Obtenir l'embedding du mot original
    original_word_embedding = model_st.encode(target_word, convert_to_tensor=True)

    case_results = []
    # Pour chaque mot prédit par le MLM, calculer sa similarité avec le mot original
    for pred in predictions:
        predicted_word = pred['token_str']
        prediction_score = pred['score']

        # Obtenir l'embedding du mot prédit
        predicted_word_embedding = model_st.encode(predicted_word, convert_to_tensor=True)

        # Calculer la similarité cosinus entre les deux embeddings.
        similarity = util.cos_sim(original_word_embedding, predicted_word_embedding).item()

        case_results.append({
            'original_sentence': sentence,
            'target_word': target_word,
            'predicted_word': predicted_word,
            'prediction_score_mlm': prediction_score,
            'semantic_similarity_score': similarity
        })

    # Trier les résultats par similarité sémantique décroissante pour ce cas
    case_results = sorted(case_results, key=lambda x: x['semantic_similarity_score'], reverse=True)

    print(f"Résultats pour '{target_word}' :")
    for res in case_results:
        print(f"- Mot: '{res['predicted_word']}', Score MLM: {res['prediction_score_mlm']:.4f}, Sim. Sémantique: {res['semantic_similarity_score']:.4f}")
    
    return case_results

# --- 5. Fonction pour sauvegarder tous les résultats dans un fichier Excel ---
def save_all_results_to_excel(all_results_list, file_path="mlm_synonym_discovery_results.xlsx"):

    try:
        df = pd.DataFrame(all_results_list)
        df.to_excel(file_path, index=False)
        print(f"\n Tous les résultats ont été sauvegardés avec succès dans : {file_path}")
    except Exception as e:
        print(f"\n Erreur lors de la sauvegarde du fichier Excel : {e}")

# --- Exécution principale ---
if __name__ == "__main__":
    all_combined_results = []

    for case in test_cases:
        sentence = case["sentence"]
        target_word = case["target_word"]
        
        # Exécuter l'analyse pour chaque cas et ajouter les résultats à la liste globale
        results_for_case = analyze_word_in_sentence(sentence, target_word, unmasker, model_st)
        all_combined_results.extend(results_for_case)
    
    # Sauvegarder tous les résultats combinés dans un seul fichier Excel
    save_all_results_to_excel(all_combined_results)