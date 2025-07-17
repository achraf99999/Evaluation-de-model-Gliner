import re, json, pathlib
from tqdm.auto import tqdm

# ------------------------------------------------------------------
# Regex that matches: words, or single non-space punctuation
_TOKEN_RE = re.compile(r"\n\n|\n|[A-Za-z0-9_]+|[^\w\s]", flags=re.UNICODE)

def tokenize(text: str):
    """Return tokens and (start_char, end_char) offsets (end exclusive)."""
    tokens, offsets = [], []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets

def char_span_to_token_span(char_span, offset_table):
    """Map a char-level (start, end) span to (start_tok, end_tok) inclusive."""
    c_start, c_end = char_span            
    start_tok = end_tok = None
    for idx, (t_start, t_end) in enumerate(offset_table):
        # First token whose start >= char_start
        if start_tok is None and t_start >= c_start:
            start_tok = idx
        # Last token whose end <= char_end
        if t_start < c_end and t_end <= c_end:
            end_tok = idx
    if start_tok is None or end_tok is None:
        raise ValueError(f"Span {char_span} does not align with tokens.")
    return start_tok, end_tok
# ------------------------------------------------------------------

SOURCE_FILE = "data.json"   
DEST_FILE   = "gliner--style.json"


import chardet
raw = pathlib.Path(SOURCE_FILE).read_bytes()
guess = chardet.detect(raw)["encoding"]
print("Best guess:", guess)
src_examples = json.loads(raw.decode(guess, errors="replace"))

dest_examples = []

for ex in tqdm(src_examples, desc="Converting examples"):
    tokens, offsets = tokenize(ex["text"])
    triples = []

    for ent in ex["entities"]:
        label = ent["text"]             
        # Some datasets list multiple (discontinuous) spans in `ent["spans"]`
        spans = (ent["spans"]
                 if isinstance(ent["spans"][0], (list, tuple))
                 else [ent["spans"]])      # normalise to list[list[int,int]]
        for c_start, c_end in spans:
            t_start, t_end = char_span_to_token_span((c_start, c_end), offsets)
            triples.append([t_start, t_end, label])

    dest_examples.append({"tokenized_text": tokens, "ner": triples})

# ------------------------------------------------------------------
pathlib.Path(DEST_FILE).write_text(
    json.dumps(dest_examples, ensure_ascii=False, indent=2),
    encoding="utf-8"      )   
    
print(f"  Wrote {len(dest_examples)} examples to {DEST_FILE}")
