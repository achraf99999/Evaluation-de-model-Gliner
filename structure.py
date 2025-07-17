import json
from collections import defaultdict

def transform_synonym_to_entity_structure(input_file, output_file):
    """
    Transforme la structure par synonymes vers la structure par entités/documents
    """
    # Charger les données du fichier source
    with open(input_file, 'r', encoding='utf-8') as f:
        synonym_data = json.load(f)
    
    # Dictionnaire pour regrouper par text_id
    documents = defaultdict(lambda: {
        'text_id': '',
        'text': '',
        'entities': []
    })
    
    # Parcourir toutes les clés de synonymes
    for synonym_key, entities_list in synonym_data.items():
        # Extraire le code d'entité (partie avant __)
        if '__' in synonym_key:
            entity_code = synonym_key.split('__')[0]
        else:
            entity_code = synonym_key
        
        # Parcourir toutes les entités de ce synonyme
        for entity_data in entities_list:
            text_id = entity_data['text_id']
            text = entity_data['text']
            span = entity_data['span']
            entity_text = entity_data['entity_text']
            
            # Initialiser le document s'il n'existe pas encore
            if not documents[text_id]['text_id']:
                documents[text_id]['text_id'] = text_id
                documents[text_id]['text'] = text
            
            # Ajouter l'entité au document
            entity_entry = {
                'code_entity': entity_code,
                'spans': span,
                'entity': entity_text
            }
            
            documents[text_id]['entities'].append(entity_entry)
    
    # Convertir en liste finale
    result = list(documents.values())
    
    # Sauvegarder dans le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f" Transformation terminée !")
    print(f" Documents traités: {len(result)}")
    
    # Statistiques
    total_entities = sum(len(doc['entities']) for doc in result)
    print(f" Total d'entités: {total_entities}")
    
    # Compter les codes d'entités uniques
    entity_codes = set()
    for doc in result:
        for entity in doc['entities']:
            entity_codes.add(entity['code_entity'])
    
    print(f"️  Codes d'entités uniques: {len(entity_codes)}")
    print(f" Liste des codes: {sorted(entity_codes)}")
    
    return result

def analyze_transformed_structure(file_path):
    """
    Analyse la structure du fichier transformé
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n=== ANALYSE DU FICHIER TRANSFORMÉ ===")
    print(f" Nombre de documents: {len(data)}")
    
    if data:
        # Analyser le premier document
        first_doc = data[0]
        print(f" Structure d'un document:")
        print(f"   - text_id: {first_doc['text_id']}")
        print(f"   - text: {len(first_doc['text'])} caractères")
        print(f"   - entities: {len(first_doc['entities'])} entités")
        
        if first_doc['entities']:
            first_entity = first_doc['entities'][0]
            print(f" Structure d'une entité:")
            print(f"   - code_entity: {first_entity['code_entity']}")
            print(f"   - spans: {first_entity['spans']}")
            print(f"   - entity: {first_entity['entity']}")
    
    # Statistiques globales
    total_entities = sum(len(doc['entities']) for doc in data)
    print(f"\n Statistiques globales:")
    print(f"   - Total d'entités: {total_entities}")
    print(f"   - Moyenne d'entités par document: {total_entities/len(data):.2f}")
    
    # Distribution des codes d'entités
    entity_code_counts = defaultdict(int)
    for doc in data:
        for entity in doc['entities']:
            entity_code_counts[entity['code_entity']] += 1
    
    print(f"\n  Distribution des codes d'entités:")
    for code, count in sorted(entity_code_counts.items()):
        print(f"   - {code}: {count} occurrences")

def validate_transformation(original_file, transformed_file):
    """
    Valide que la transformation a conservé toutes les données
    """
    # Charger les fichiers
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(transformed_file, 'r', encoding='utf-8') as f:
        transformed_data = json.load(f)
    
    # Compter les entités dans l'original
    original_count = sum(len(entities) for entities in original_data.values())
    
    # Compter les entités dans le transformé
    transformed_count = sum(len(doc['entities']) for doc in transformed_data)
    
    print(f"\n=== VALIDATION DE LA TRANSFORMATION ===")
    print(f" Entités dans l'original: {original_count}")
    print(f" Entités après transformation: {transformed_count}")
    
    if original_count == transformed_count:
        print(" Transformation réussie ! Toutes les entités ont été conservées.")
    else:
        print(" Attention ! Certaines entités ont été perdues lors de la transformation.")
    
    return original_count == transformed_count

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemins des fichiers
    input_file = r"C:\Users\21650\Desktop\pfe\outputs\debug_by_synonym.json"
    output_file = r"C:\Users\21650\Desktop\pfe\outputs\debug_by_synonym...json"
    
    print(" Début de la transformation...")
    
    # Transformer la structure
    result = transform_synonym_to_entity_structure(input_file, output_file)
    
    # Analyser le résultat
    analyze_transformed_structure(output_file)
    
    # Valider la transformation
    validate_transformation(input_file, output_file)
    
    print(f"\n Fichier de sortie sauvegardé: {output_file}")