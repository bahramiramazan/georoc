import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math

import torch.nn.functional as F






class georoc_model(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        embeddings

    ):

        super().__init__()


        vocab_size, d_model = embeddings.size()

        filter_sizes = [1,2,3,5]
        num_filters = 36
        embed_size=d_model
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)


        self.classifier = nn.Linear(144, 2)
        self.d_model = d_model

        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x=x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x) 


        x = self.classifier(x)

        return x