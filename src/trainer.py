# Trainer


import torch.nn as nn
import torch
# import torch.nn.functional as F
from transformers import LayoutLMv3ForTokenClassification
import torch.nn.functional as nnf


def loss_fn(pred,target):
    return nn.CrossEntropyLoss()(pred.view(-1,4),target.view(-1))


class ModelModule(nn.Module):
    def __init__(self,n_classes:int) -> None:
        super().__init__()
        self.model = LayoutLMv3ForTokenClassification.from_pretrained('../inputs/layoutlmv3Microsoft')
        self.cls_layer = nn.Sequential(nn.Linear(in_features = 2,
                                                out_features = 512),
                                      nn.ReLU(), nn.Linear(in_features = 512, out_features = n_classes))

    def forward(self,input_ids,attention_mask,bbox,pixel_values,lables=None):
        output = self.model(input_ids,attention_mask=attention_mask,bbox=bbox,pixel_values=pixel_values)

        output = self.cls_layer(output.logits)

        prob = nnf.softmax(output, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)

        print("Probability score :", prob)
        print("top_p, top_class ",top_p, top_class)
        loss = loss_fn(output,lables)

        return  output, loss
