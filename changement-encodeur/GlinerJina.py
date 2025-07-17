import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os


# Span representation module (used to embed entity spans)
class SpanMarkerV0(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.project_start = nn.Sequential(
            nn.Linear(hidden_size, 3072),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(3072, hidden_size)
        )
        self.project_end = nn.Sequential(
            nn.Linear(hidden_size, 3072),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(3072, hidden_size)
        )
        self.out_project = nn.Sequential(
            nn.Linear(hidden_size * 2, 3072),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(3072, hidden_size)
        )

    def forward(self, start_emb, end_emb):
        h_start = self.project_start(start_emb)
        h_end = self.project_end(end_emb)
        return self.out_project(torch.cat([h_start, h_end], dim=-1))


# Manual implementation of GLiNER with Jina Embeddings replacing label encoder
class ManualGLiNER(nn.Module):
    def __init__(self, text_encoder_name="microsoft/deberta-v3-large", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        def load_weights(name, strip_model_prefix=False):
            path = os.path.join("weightslarge", name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            obj = torch.load(path, map_location=self.device, weights_only=True)
            if isinstance(obj, dict):
                if strip_model_prefix:
                    obj = {k.replace("model.", "", 1): v for k, v in obj.items() if k.startswith("model.")}
                return obj
            return obj

        # Text encoder (DeBERTa large)
        config = DebertaV2Config.from_pretrained(text_encoder_name)
        self.encoder = DebertaV2Model(config).to(self.device)
        self.encoder.load_state_dict(load_weights("gliner_deberta_encoder_weights.pt", strip_model_prefix=False))

        # Projection layer: 1024 -> 768
        self.projection = nn.Linear(1024, 768).to(self.device)
        self.projection.load_state_dict(load_weights("gliner_projection_weights.pt"))

        # Tokenizer for input text
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

        # Span encoder
        self.span_rep_layer = SpanMarkerV0().to(self.device)
        self.span_rep_layer.load_state_dict(load_weights("gliner_span_rep_layer_weights.pt"))

        # Prompt layer
        self.prompt_rep_layer = nn.Sequential(
            nn.Linear(768, 3072),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(3072, 768)
        ).to(self.device)
        self.prompt_rep_layer.load_state_dict(load_weights("gliner_prompt_rep_layer_weights.pt"))

        # Jina Embeddings model to replace label encoder
        self.label_encoder = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, truncate_dim=768)

        # Pooler to transform embeddings
        self.pooler = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
        ).to(self.device)
        self.pooler[0].load_state_dict(load_weights("gliner_label_pooler.pt"))

        # RNN encoder for label embeddings
        self.label_rnn = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True).to(self.device)
        self.label_rnn.load_state_dict(load_weights("gliner_label_rnn.pt"))

    def generate_spans(self, seq_len, max_length=10):
        return [(start, end) for start in range(seq_len) for end in range(start, min(seq_len, start + max_length))]

    def predict(self, text, labels, threshold=0.3, max_span_length=10):
        print("\n=== Step 1: Text Encoding ===")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.encoder(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)
        print(f"Text hidden shape: {hidden.shape}")
        hidden = self.projection(hidden)
        print(f"Projected hidden shape: {hidden.shape}")

        print("\n=== Step 2: Label Encoding ===")
        task = "text-matching"
        label_embeds = self.label_encoder.encode(labels, task=task, prompt_name=task, convert_to_tensor=True).to(torch.float32).unsqueeze(0).to(self.device)
        print(f"Raw label embeddings shape: {label_embeds.shape}")

        pooled = self.pooler(label_embeds)
        print(f"Pooled label embeddings shape: {pooled.shape}")

        rnn_out, _ = self.label_rnn(pooled)
        print(f"LSTM output shape: {rnn_out.shape}")
        label_vectors = self.prompt_rep_layer(rnn_out.squeeze(0))
        print(f"Label vectors shape after prompt layer: {label_vectors.shape}")

        print("\n=== Step 3: Span Generation ===")
        spans = self.generate_spans(hidden.size(0), max_span_length)
        print(f"Generated {len(spans)} spans")

        span_vectors = []
        for idx, (start, end) in enumerate(spans):
            start_vec = hidden[start]
            end_vec = hidden[end]
            span_embed = self.span_rep_layer(start_vec.unsqueeze(0), end_vec.unsqueeze(0))
            span_vectors.append(span_embed)
            if idx < 1:
                print(f"Span {start}-{end} embedding shape: {span_embed.shape}")

        span_vectors = torch.cat(span_vectors, dim=0)
        print(f"All span vectors shape: {span_vectors.shape}")

        print("\n=== Step 4: Similarity Computation ===")
        scores = torch.matmul(span_vectors, label_vectors.T)
        print(f"Similarity scores shape: {scores.shape}")
        probs = torch.sigmoid(scores)
        print(f"Example probabilities (first 5):\n{probs[:5]}")

        print("\n=== Step 5: Threshold Filtering ===")
        predictions = []
        for i, (start, end) in enumerate(spans):
            for j, label in enumerate(labels):
                score = probs[i, j].item()
                if score >= threshold:
                    token_span = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end + 1])
                    span_text = self.tokenizer.convert_tokens_to_string(token_span)
                    prediction = {
                        "text": span_text,
                        "start": start,
                        "end": end,
                        "label": label,
                        "score": round(score, 4)
                    }
                    print(f"[{label} @ {start}-{end}] → {span_text} (score={score:.4f})")
                    predictions.append(prediction)

        print(f"\n✅ Total predictions: {len(predictions)}")
        return predictions