from transformers import AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# iterate / inspect / save weights directly on `model`
for name, param in model.named_parameters():
    print(name, param.size())

