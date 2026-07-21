import torch.nn as nn
from transformers import BertModel
import config


class ruBERT(nn.Module):
    def __init__(self, model_name=config.BERT_MODEL_NAME, num_classes : int=config.NUM_CLASSES, dropout : int =0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        x = self.classifier(x)
        return x