import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

class GemmaDetector(nn.Module):
    def __init__(self, embedding_dim, prj_dim):
        super().__init__()
        self.under_prj = nn.Linear(2 * embedding_dim, prj_dim)
        self.up_prj = nn.Linear(prj_dim, embedding_dim)
        self.prediction_layer = nn.Linear(embedding_dim, 2)
    def forward(self, encoded_context, encoded_sentences, labels):
        x = torch.hstack((encoded_context, encoded_sentences))
        x = self.under_prj(x)
        x = self.up_prj(x)
        x = nn.functional.relu(x)
        logits = self.prediction_layer(x)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return SequenceClassifierOutput(logits=logits, loss=loss)