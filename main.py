
import os
from os.path import exists
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

import math

import torch
import torch.nn as nn
import torchtext

import argparse
from georoc_util import *


from georoc_model import *
# from matplotlib import pyplot as plt
import pydoi
import json
import pandas as pd


import csv


device = torch.device("mps" if torch.cuda.is_available() else "cpu")



def get_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--task", help=" data_collection,proprocess, train , eval")

    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.name=%s" % args.task)
    return args.task





def georoc_collect_data():
    my_api_key="78c563ff10f1d9edbbbf4e0a6fc1e148"

    i=0
    count=0
    clms=["CITATIONS","DOI","AUTHORS","YEAR","TITLE","JOURNAL","VOL","ISSUE","PAGES","BOOK_TITLE","EDITOR","PUBLISHER","FORMATTED_CITATION","ABSTRACT"]
    df = pd.DataFrame(columns = clms)
    with open("data/2022-12-4EZ7ID_CITATION.tab") as fruits_file:
        tsv_reader = csv.reader(fruits_file, delimiter="\t")

        # Skip the first row, which is the header
        next(tsv_reader)
        
        for row in tsv_reader:
            
            i=i+1
            if i<(468+318+231+169+64+98+75+63+42+30+21+88+180+550+280+125+1724+619+438+379+531+1160+1077+1055+1000+300):
                continue
            (CITATIONS,DOI,AUTHORS,YEAR,TITLE,JOURNAL,VOL,ISSUE,PAGES,BOOK_TITLE,EDITOR,PUBLISHER,FORMATTED_CITATION) = row

            url=pydoi.resolve(DOI)

            if url:
                if "values" in url.keys():

                    url=url["values"][0]
                    if "data" in url.keys():
                        url=url["data"]['value']
                        if "elsevier" in url:
                            
                            y = scopus_paper_date(DOI,my_api_key)
                            count=count+1
                            print(count)
                            if y==False:
                                continue
                      
                            # Parse document
                            

                            json_acceptable_string = y.text
                            d = json.loads(json_acceptable_string)
                            # Print document
                            if  (d.get('full-text-retrieval-response') is None):
                                continue
                            ABSTRACT=d['full-text-retrieval-response']['coredata']['dc:description']
                         
                            abstract=str(ABSTRACT).replace(",", "")
    #                         abstract=abstract.translate(str.maketrans('', '', string.punctuation))
                         
                            dataframe=[
                             CITATIONS,
                                DOI,
                                AUTHORS,
                                YEAR,
                                TITLE,
                                JOURNAL,
                               VOL,
                                ISSUE,
                                PAGES,
                               BOOK_TITLE,
                                EDITOR,
                                PUBLISHER,
                                FORMATTED_CITATION,
                               abstract
                
                            ]

                            df.loc[count] = dataframe

                            df.to_csv("data/data.csv")
         
def georoc_eval(model,data):

    checkpoint = torch.load('checkpoint.t7')
    model.load_state_dict(checkpoint['state_dict'])

    PREDICTIONS=[]
    Y=[]
    n_epochs=1
    i=0

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        model.eval()
    
        for x in data:

            src=torch.cat((x[0], x[1]), 1)
            src_mask = (src != 2) 
            
            
            citation=x[2].data[0]
            y = x[3]
            y=list(y)
           
            y=torch.tensor([x.item() for x in y])

            predictions = model(src.to(device))
            labels = y.to(device) 

            
            correct = predictions.argmax(axis=1) == labels
            correct_tmp,total_tmp = correct.sum().item() , correct.size(0)
            notcorrect_tmp=correct.size(0)-correct.sum().item()

            PREDICTIONS.extend(predictions.argmax(axis=1).tolist())
            Y.extend(labels.tolist())
            i=i+1

        tp,tn,fp,fn=0,0,0,0
        for i in range(len(Y)):
            if Y[i]==0:
                if PREDICTIONS[i]==0:
                    tn=tn+1
                else:
                    fp=fp+1
            if Y[i]==1:
                if PREDICTIONS[i]==1:
                    tp=tp+1
                else:
                    fn=fn+1
            
            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

        acc=(tp+tn)/(tp+tn+fp+fn) 
        tp_rate=(tp)/(tp+fn)
        fn_rate=fn/(tp+fn)
        tn_rate=tn/(tn+fp)
        fp_rate=fp/(tn+fp)
        

        print(acc,)
        print('acc,',acc)
        print('confusion matrix',tp,tn,fp,fn)
        print('tp_rate=(tp)/(tp+fn)',tp_rate)
        print('fn_rate=tp/(tp+fn)',fn_rate)
        print('tn_rate=tn/(tn+fp)',tn_rate)
        print('fp_rate=fp/(tn+fp)',fp_rate)
        
def georoc_train_eval(mode='train'):
    args=DictX(Args)
    train_loader,dev_loader=georoc_data(args)



    DATA=get_abstracts()
    meta_data=DATA
    data=DATA

    epochs = 50
    word_vectors = torch_from_json(args.word_emb_file)
    model = georoc_model(
        word_vectors 
    ).to(device)

    if mode=='train':
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )

    torch.manual_seed(0)

    
    DATA=meta_data

    mineral_set,mineral_set_L=get_minerals(DATA)

    if mode =='eval':
        georoc_eval(model,dev_loader)
        


    if mode=='train':
        data=train_loader#train_loader
        LOSS=[]
        ACC=[]
        n_epochs=4

        for epoch in range(n_epochs):
                epoch_loss = 0
                epoch_correct = 0
                epoch_count = 0
                model.train()
            
                for x in data:
        
                    src=torch.cat((x[0], x[1]), 1)
                    src_mask = (src != 2) 
                    
                    
                    citation=x[2].data[0]
                    y = x[3]
                    y=list(y)
                   
                    y=torch.tensor([x.item() for x in y])

                    predictions = model(src.to(device))
                    labels = y.to(device) 

                    loss = criterion(predictions, labels)
                    LOSS.append(loss.item())

                    correct = predictions.argmax(axis=1) == labels
                    acc = correct.sum().item() / correct.size(0)
                    ACC.append(acc)

                    epoch_correct += correct.sum().item()
                    epoch_count += correct.size(0)

                    epoch_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                    optimizer.step()
                 
        
                
                if mode=='train':
                    state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                    }
                    savepath='checkpoint.t7'
                    torch.save(state,savepath)
                georoc_eval(model,dev_loader)


                  

def georoc_pre_process():
    pre_process()



if __name__ == "__main__":
    # python main.py  --task train/eval/preprocess/collect
    t=get_args()
    print("task is :",t)

    if str(t) =="train":
        print("train")
        georoc_train_eval()
    if str(t) =="eval":
        print("eval")
        georoc_train_eval('eval')

    if str(t)=='preprocess' :
        georoc_pre_process()
    if str(t)=='collect' :
        georoc_collect_data() 
    else:
        print('task could be one of the  train or eval or preprocess or collect')
    


  
