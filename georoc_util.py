import pandas as pd 

import torch 
import numpy as np
#

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from collections import OrderedDict
from json import dumps

from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load


import logging
import os
import queue
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import ujson as json

from collections import Counter

from georoc_data import *

import httpx
import time



def scopus_paper_date(paper_doi,apikey):
    reply=True
    apikey=apikey
    headers={
        "X-ELS-APIKey":apikey,
        "Accept":'application/json'
         }
    timeout = httpx.Timeout(10.0, connect=60.0)
    client = httpx.Client(timeout=timeout,headers=headers)
    
  
    query="&view=FULL"
    url=f"https://api.elsevier.com/content/article/doi/"+paper_doi
    r=client.get(url)
    try:
        response = r
        response.raise_for_status()
    except httpx.HTTPError as ex:
        Reply=False
    if reply:
        return r
    else:
        return False


def torch_from_json(path, dtype=torch.float32):
    """
    # adapted from "https://github.com/minggg/squad/blob/master/"
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor

def georoc_data(args):
	    # args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    # log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # Get embeddings

    word_vectors = torch_from_json(args.word_emb_file)


    print(word_vectors.shape)


    train_dataset = GEOROC(args.train_record_file)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = GEOROC(args.dev_record_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                     collate_fn=collate_fn)

    return train_loader,dev_loader

def encoder_fun(mineral,mineral_set):
    encoded_mineral=torch.zeros(1,len(mineral_set))

    mineral_set_L=list(mineral_set)
    mineral=mineral.split()
    for m in mineral:
        i=mineral_set_L.index(m)
        encoded_mineral[0][i]=1
    return encoded_mineral



def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x



def class_data(DATA):

	mineral_set=set()
	for r in DATA["MINERAL"].unique():
	 
	    for m in str(r).split():
	        mineral_set.add(str(m).lower())
	mineral_set_L=list(mineral_set)

def get_abstracts():
    meta_data = pd.read_csv('data//2022-12-4EZ7ID_METADATA.csv',low_memory=False)
    DATA=meta_data
    # DATA["MINERAL"]=meta_data["MINERAL"].str.replace(";", "")
    DATA["MINERAL"]=DATA["MINERAL"].str.strip(";")
    DATA["MINERAL"]=DATA["MINERAL"].str.replace("+", " ",regex=True)
    DATA["MINERAL"]=DATA["MINERAL"].str.replace("/", " ",regex=True)
    DATA["MINERAL"]=DATA["MINERAL"].str.replace("-", " ",regex=True)
    DATA["MINERAL"]=DATA["MINERAL"].str.replace("_", " ",regex=True)
    DATA["MINERAL"]=DATA["MINERAL"].str.replace(";", " ",regex=True)
    DATA["MINERAL"]=DATA["MINERAL"].str.strip()
    DATA[["MINERAL"]]
    return DATA

def get_minerals(DATA):
    mineral_set=set()

    for r in DATA["MINERAL"].unique():
     
        for m in str(r).split():
            mineral_set.add(str(m).lower())

    mineral_set_L=list(mineral_set)

    return  mineral_set,mineral_set_L

 
      