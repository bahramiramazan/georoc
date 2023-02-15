import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from georoc_layers import *
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
        self.generator = generator
#         self.deep= Deep()

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
#         print("memory shape, tgt shape",memory.shape,tgt.shape)
#         print("type",type(tgt_mask),type(tgt))
        tgt=tgt.reshape(tgt.shape[0],tgt.shape[1],1)
#         print(self.deep(tgt).shape)
        return self.decoder(tgt, memory, src_mask, tgt_mask)






class Net(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        embeddings,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
        dropout=0.1,
        activation="relu",
        classifier_dropout=0.1,
    ):

        super().__init__()

        vocab_size, d_model = embeddings.size()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        filter_sizes = [1,2,3,5]
        num_filters = 36
        embed_size=d_model


        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        #592  self.classifier = nn.Linear(d_model, 2)
        self.classifier = nn.Linear(144, 2)
        self.d_model = d_model

        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        # x = self.pos_encoder(x)
        # x = self.transformer_encoder(x)

        # shape before torch.Size([1, 1, 750, 300])
        # x .shape before conv torch.Size([10, 622, 300])
        x=x.reshape(x.shape[0],1,x.shape[1],x.shape[2])


        print("x .shape before conv",x.shape)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x) 
        print("x .shape",x.shape)

        # x = x.mean(dim=1)

        x = self.classifier(x)

        return x