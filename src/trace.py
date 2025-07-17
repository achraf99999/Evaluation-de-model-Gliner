#!/usr/bin/env python

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ‑‑‑ Project‑specific imports
from src.config import (
    ENTITY_TYPES,            
    OUTPUT_DIR,
    DEFAULT_JACCARD_THRESHOLD,
)
from src.utils import load_json, jaccard

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

JACCARD_THRESHOLD: float = DEFAULT_JACCARD_THRESHOLD
Span = Tuple[int, int]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span_to_set(span: Span) -> set:
    return set(range(span[0], span[1]))


def _debug_matching(text_id: str, entity_code: str, gold_entities: List[dict], pred_entities: List[dict]):
    logger.info(f"\n=== DEBUG MATCHING for {text_id} - {entity_code} ===")
    logger.info(f"Gold entities: {len(gold_entities)}")
    for g in gold_entities:
        logger.info(f"  Gold: {g['spans']} -> '{g['entity']}'")
    
    logger.info(f"Predicted entities: {len(pred_entities)}")
    for p in pred_entities:
        logger.info(f"  Pred: {p['spans']} -> '{p['entity']}'")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_all_entity_traces(
    predictions: List[dict],
    ground_truth: List[dict],
    jaccard_threshold: float = JACCARD_THRESHOLD,
):
    """
    Évalue les performances en comparant prédictions et vérité terrain.
    
    Args:
        predictions: Liste des documents avec prédictions
        ground_truth: Liste des documents avec annotations gold
        jaccard_threshold: Seuil pour considérer un match partiel
    
    Returns:
        List[dict]: Liste des traces d'évaluation
    """
    traces: List[dict] = []
    
    # Indexer les documents par text_id
    pred_by_id = {str(doc["text_id"]).strip(): doc for doc in predictions}
    gold_by_id = {str(doc["text_id"]).strip(): doc for doc in ground_truth}
    
    logger.info("Predictions loaded: %d documents", len(pred_by_id))
    logger.info("Ground truth loaded: %d documents", len(gold_by_id))
    
    # Obtenir tous les codes d'entités présents
    all_entity_codes = set()
    for doc in ground_truth:
        for ent in doc["entities"]:
            all_entity_codes.add(ent["code_entity"])
    
    logger.info(f"Entity codes found: {sorted(all_entity_codes)}")
    
    # Traiter chaque document
    all_text_ids = set(gold_by_id.keys()) | set(pred_by_id.keys())
    
    for text_id in all_text_ids:
        gold_doc = gold_by_id.get(text_id)
        pred_doc = pred_by_id.get(text_id)
        
        if not gold_doc:
            logger.warning(f"No ground truth for text_id: {text_id}")
            continue
            
        if not pred_doc:
            logger.warning(f"No predictions for text_id: {text_id}")
            # Toutes les entités gold deviennent FN
            for ent in gold_doc["entities"]:
                span = tuple(ent["spans"])
                traces.append({
                    "text_id": text_id,
                    "status": "FN",
                    "span": list(span),
                    "entity_text": gold_doc["text"][span[0]:span[1]],
                    "gold_entity": ent["entity"],
                    "entity_code": ent["code_entity"],
                })
            continue
        
        # Traiter chaque type d'entité séparément
        for entity_code in all_entity_codes:
            
            gold_entities = [
                ent for ent in gold_doc["entities"] 
                if ent["code_entity"] == entity_code
            ]
            pred_entities = [
                ent for ent in pred_doc.get("entities", []) 
                if ent["code_entity"] == entity_code
            ]
            
            if not gold_entities and not pred_entities:
                continue
                

            
            # Créer des maps pour faciliter les correspondances
            gold_map = {tuple(g["spans"]): g for g in gold_entities}
            pred_map = {tuple(p["spans"]): p for p in pred_entities}
            
            # Obtenir les spans d'autres types d'entités (pour TN)
            other_gold_spans = [
                tuple(ent["spans"]) for ent in gold_doc["entities"] 
                if ent["code_entity"] != entity_code
            ]
            
            matched_gold, matched_pred = set(), set()
            
            # Phase 1 – Exact TP
            exact_matches = set(gold_map.keys()) & set(pred_map.keys())
            for span in exact_matches:
                matched_gold.add(span)
                matched_pred.add(span)
                
                p = pred_map[span]
                g = gold_map[span]
                
                traces.append({
                    "text_id": text_id,
                    "status": "TP",
                    "match_type": "exact",
                    "span": list(span),
                    "entity_text": gold_doc["text"][span[0]:span[1]],
                    "gold_entity": g["entity"],
                    "predicted_text": pred_doc["text"][span[0]:span[1]],
                    "predicted_entity": p["entity"],
                    "entity_code": entity_code,
                    "score": p.get("score"),
                })
                
                logger.debug(f"TP EXACT: {text_id} - {span} - '{g['entity']}'")
            
            # Phase 2 – Partial TP (Jaccard >= threshold)
            unmatched_pred = [s for s in pred_map.keys() if s not in matched_pred]
            unmatched_gold = [s for s in gold_map.keys() if s not in matched_gold]
            
            for p_span in unmatched_pred:
                p_set = _span_to_set(p_span)
                best_span, best_score = None, 0.0
                
                for g_span in unmatched_gold:
                    score = jaccard(p_set, _span_to_set(g_span))
                    if score >= jaccard_threshold and score > best_score:
                        best_span, best_score = g_span, score
                
                if best_span is not None:
                    matched_pred.add(p_span)
                    matched_gold.add(best_span)
                    unmatched_gold.remove(best_span)
                    
                    p = pred_map[p_span]
                    g = gold_map[best_span]
                    
                    traces.append({
                        "text_id": text_id,
                        "status": "TP",
                        "match_type": "partial",
                        "span": list(best_span),
                        "entity_text": gold_doc["text"][best_span[0]:best_span[1]],
                        "gold_entity": g["entity"],
                        "predicted_span": list(p_span),
                        "predicted_text": pred_doc["text"][p_span[0]:p_span[1]],
                        "predicted_entity": p["entity"],
                        "jaccard_score": round(best_score, 4),
                        "entity_code": entity_code,
                        "score": p.get("score"),
                    })
                    
                    logger.debug(f"TP PARTIAL: {text_id} - {best_span} - '{g['entity']}' (Jaccard: {best_score:.3f})")
            
            # Phase 3 – FP (predictions not matched)
            for span, p in pred_map.items():
                if span not in matched_pred:
                    traces.append({
                        "text_id": text_id,
                        "status": "FP",
                        "span": list(span),
                        "entity_text": pred_doc["text"][span[0]:span[1]],
                        "predicted_entity": p["entity"],
                        "entity_code": entity_code,
                        "score": p.get("score"),
                    })
                    
                    logger.debug(f"FP: {text_id} - {span} - '{p['entity']}'")
            
            # Phase 4 – FN (gold entities not matched)
            for span, g in gold_map.items():
                if span not in matched_gold:
                    traces.append({
                        "text_id": text_id,
                        "status": "FN",
                        "span": list(span),
                        "entity_text": gold_doc["text"][span[0]:span[1]],
                        "gold_entity": g["entity"],
                        "entity_code": entity_code,
                    })
                    
                    logger.debug(f"FN: {text_id} - {span} - '{g['entity']}'")
            
            # Phase 5 – TN (count only)
            if other_gold_spans:
                pred_sets = [_span_to_set(s) for s in pred_map.keys()]
                tn_count = sum(
                    1 for other_span in other_gold_spans
                    if not any(_span_to_set(other_span) & ps for ps in pred_sets)
                )
                if tn_count > 0:
                    traces.append({
                        "text_id": text_id,
                        "status": "TN",
                        "entity_code": entity_code,
                        "count": tn_count,
                    })
    
    return traces




# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    logger.info("Loading resources …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    predictions_path = Path(r"C:\Users\21650\Desktop\pfe\outputs\debug_by_synonym...json")
    ground_truth_path = Path(r"C:\Users\21650\Desktop\pfe\data\fulldata.json")

    try:
        # Charger les données brutes sans restrictions
        raw_predictions = load_json(predictions_path)
        raw_ground_truth = load_json(ground_truth_path)
        
        # Convertir les données au format attendu
        if isinstance(raw_predictions, dict):
            # Cas 1: Dictionnaire où les valeurs sont des documents
            predictions = list(raw_predictions.values())
        elif isinstance(raw_predictions, list):
            predictions = raw_predictions
        else:
            raise ValueError(f"Format de prédictions inattendu: {type(raw_predictions)}")
        
        if isinstance(raw_ground_truth, dict):
            ground_truth = list(raw_ground_truth.values())
        elif isinstance(raw_ground_truth, list):
            ground_truth = raw_ground_truth
        else:
            raise ValueError(f"Format de vérité terrain inattendu: {type(raw_ground_truth)}")
        
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return

    # Diagnostic
    logger.info(f"Type de prédictions après conversion: {type(predictions)}")
    logger.info(f"Nombre de documents prédits: {len(predictions)}")
    logger.info(f"Type de vérité terrain après conversion: {type(ground_truth)}")
    logger.info(f"Nombre de documents de vérité terrain: {len(ground_truth)}")
    
    # Vérification du premier document
    if predictions:
        first_pred = predictions[0]
        logger.info(f"Type du premier document prédit: {type(first_pred)}")
        if isinstance(first_pred, dict):
            logger.info(f"Clés du premier document: {list(first_pred.keys())}")
        else:
            logger.error("Le premier document prédit n'est pas un dictionnaire")
            return
    
    # Le reste du code inchangé...
    traces = extract_all_entity_traces(predictions, ground_truth)
        
    # Sauvegarder les traces
    out_dir = OUTPUT_DIR / "entity_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "evaluation_tracesNew.json"
    
    with open(traces_path, "w", encoding="utf-8") as fh:
        json.dump(traces, fh, ensure_ascii=False, indent=2)
    
 
    
    logger.info(f"Traces saved to {traces_path}")


if __name__ == "__main__":
    main()