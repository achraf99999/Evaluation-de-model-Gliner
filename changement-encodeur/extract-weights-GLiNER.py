from gliner import GLiNER
import torch
import os

# Create the output directory for weights
output_dir = "weightslarge"
os.makedirs(output_dir, exist_ok=True)

# Load the large GLiNER model
model = GLiNER.from_pretrained("knowledgator/gliner-bi-large-v1.0")
base_model = model.model

# Save encoder (DeBERTa) weights
torch.save(base_model.token_rep_layer.bert_layer.model.state_dict(), os.path.join(output_dir, "gliner_deberta_encoder_weights.pt"))

# Save projection layer after encoder
torch.save(base_model.token_rep_layer.projection.state_dict(), os.path.join(output_dir, "gliner_projection_weights.pt"))

# Save pooler from label encoder
pooler_weights = {
    "weight": base_model.token_rep_layer.labels_encoder.model.pooler.dense.weight,
    "bias": base_model.token_rep_layer.labels_encoder.model.pooler.dense.bias
}
torch.save(pooler_weights, os.path.join(output_dir, "gliner_label_pooler.pt"))

# Save span representation layer
torch.save(base_model.span_rep_layer.span_rep_layer.state_dict(), os.path.join(output_dir, "gliner_span_rep_layer_weights.pt"))

# Save prompt representation layer
torch.save(base_model.prompt_rep_layer.state_dict(), os.path.join(output_dir, "gliner_prompt_rep_layer_weights.pt"))

"✅ All specified weights have been extracted and saved into 'weightslarge'"
