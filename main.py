
import argparse
from georoc_util import *
from georoc_layers import *

import util

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

import math

import torch
import torch.nn as nn

import torchtext

from georoc_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--task", help=" data_collection,proprocess, train , test")

    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.name=%s" % args.task)
    return args.task

def georoc_train():
    args=DictX(Args)
    train_loader,dev_loader=georoc_data(args)

    meta_data = pd.read_csv('data//2022-12-4EZ7ID_METADATA.csv',low_memory=False)
    DATA=meta_data
    # DATA["MINERAL"]=meta_data["MINERAL"].str.replace(";", "")
    DATA["MINERAL"]=meta_data["MINERAL"].str.strip(";")
    DATA["MINERAL"]=meta_data["MINERAL"].str.replace("+", " ")
    DATA["MINERAL"]=meta_data["MINERAL"].str.replace("/", " ")
    DATA["MINERAL"]=meta_data["MINERAL"].str.replace("-", " ")
    DATA["MINERAL"]=meta_data["MINERAL"].str.replace("_", " ")
    DATA["MINERAL"]=meta_data["MINERAL"].str.replace(";", " ")
    DATA["MINERAL"]=meta_data["MINERAL"].str.strip()
    DATA[["MINERAL"]]
    # .split(",")
    # print(list(DATA["MINERAL"].unique()))

    meta_data=DATA
    data=DATA

    epochs = 50
    word_vectors = util.torch_from_json(args.word_emb_file)
    model = Net(
        word_vectors , #TEXT.vocab.vectors,
        nhead=5,  # the number of heads in the multiheadattention models
        dim_feedforward=50,  # the dimension of the feedforward network model in nn.TransformerEncoder
        num_layers=6,
        dropout=0.0,
        classifier_dropout=0.0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    lr = 1e-4
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    torch.manual_seed(0)

    mineral_set=set()
    DATA=meta_data
    for r in DATA["MINERAL"].unique():
     
        for m in str(r).split():
            mineral_set.add(str(m).lower())
    print(mineral_set)
    mineral_set_L=list(mineral_set)



    nan=torch.tensor([[1, 0, 0]])
    j=0
    t,f=0,0
    n_epochs=3

    row_class = pd.read_csv("data/row_classs.csv")
    row_class=row_class[["row_class"]]
    
    for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_count = 0
            model.train()
        
            for x in train_loader:
    
                src=torch.cat((x[0], x[1]), 1)
                src_mask = (src != 2) 
                
                
                citation=x[2].data[0]
                y = x[3]
                y=list(y)
               
                y=torch.tensor([x.item() for x in y])
         


     
                


                # d=Batch(src,y)
                # src,tgt,src_mask,tgt_mask,tgt_y=d.src,d.tgt,d.src_mask,d.tgt_mask,d.tgt_y


                predictions = model(src.to(device))
                labels = y.to(device) 

                loss = criterion(predictions, labels)

                correct = predictions.argmax(axis=1) == labels
                acc = correct.sum().item() / correct.size(0)

                epoch_correct += correct.sum().item()
                epoch_count += correct.size(0)

                epoch_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()
             
                # if y.item()==1:
                print(loss.item(),acc)
                print("y and predicted:" )

                print(y)
                print(torch.argmax(predictions,dim=1) )
                
         

         



def georoc_pre_process():
    pre_process()








if __name__ == "__main__":
    t=get_args()
    print("task is :",t)

    if str(t) =="train":
        print("train")
        georoc_train()

    else :
        georoc_pre_process()
    


  
