import json
import logging
from collections import defaultdict
from pathlib import Path

# --- Project-specific imports (assuming these are available in your environment)
from config import OUTPUT_DIR # Make sure OUTPUT_DIR is correctly defined in your config.py

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def filter_traces_by_status(input_file: Path, output_dir: Path, statuses_to_keep: list) -> None:
    """
    Reads reorganized entity traces from an input JSON file,
    filters them to keep only specified statuses (e.g., 'FN', 'FP'),
    and saves the filtered data to a new JSON file.
    """
    logger.info(f"Loading reorganized traces from {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as fh:
            traces_by_text_id = json.load(fh)
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {input_file}. Check file integrity.")
        return

    logger.info(f"Filtering traces to keep only statuses: {statuses_to_keep}...")
    filtered_traces_by_text_id = defaultdict(list)
    total_original_traces = 0
    total_filtered_traces = 0

    for text_id, traces_list in traces_by_text_id.items():
        total_original_traces += len(traces_list)
        for trace in traces_list:
            if trace.get("status") in statuses_to_keep:
                # For TN, we save the count directly if it's the only info.
                # For FN/FP, we want the full trace.
                # If TN is included in statuses_to_keep, ensure it's handled.
                if trace.get("status") == "TN" and "count" in trace:
                    # If you truly want TN in the filtered list, and it's just a count.
                    # For FN/FP, the 'count' key is not usually present for individual traces.
                    filtered_traces_by_text_id[text_id].append(trace)
                    total_filtered_traces += trace.get("count", 1) # Add count for TN, 1 for others
                else:
                    filtered_traces_by_text_id[text_id].append(trace)
                    total_filtered_traces += 1 # Count individual FN/FP

    logger.info(f"Original total traces: {total_original_traces}")
    logger.info(f"Filtered total traces ({', '.join(statuses_to_keep)}): {total_filtered_traces}")


    # Construct output file name based on filtered statuses
    status_suffix = "_".join(statuses_to_keep).lower()
    output_filename = f"entity_evaluation_traces_{status_suffix}_by_text_idnew.json"
    output_file = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving filtered traces to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(filtered_traces_by_text_id, fh, ensure_ascii=False, indent=2)

    logger.info("Filtering complete.")
    logger.info(f"Filtered traces saved to {output_file}")


if __name__ == "__main__":
    # Define the path to the output directory and the traces file
    # This assumes 'OUTPUT_DIR' is defined in your 'config.py'.
    TRACES_DIR = "entity_traces"
    INPUT_TRACES_BY_TEXT_ID_FILE = TRACES_DIR / "all_entity_evaluation_traces.json"

    # Define the statuses you want to keep
    # For FN and FP only:
    STATUSES_TO_KEEP = ["FN", "FP"]

    filter_traces_by_status(INPUT_TRACES_BY_TEXT_ID_FILE, TRACES_DIR, STATUSES_TO_KEEP)