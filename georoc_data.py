import torch 

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from ujson import load as json_load
import os
import re
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import ujson as json
from collections import Counter
import numpy as np
import os
import spacy
import ujson as json

from args import get_setup_args
from codecs import open
from collections import Counter
import pandas as pd


nlp = spacy.blank("en")



Args= {
    "batch_size": 10,
    "char_emb_file": "./data/char_emb.json",
    "dev_eval_file": "./data/dev_eval.json",
    "dev_record_file": "./data/dev.npz",
    "drop_prob": 0.2,
    "ema_decay": 0.999,
    "eval_steps": 50000,
    "gpu_ids": [],
    "hidden_size": 512,
    "l2_wd": 0,
    "load_path": None,
    "lr": 0.5,
    "name": "baseline",
    "max_grad_norm": 5.0,
 
    "num_workers": 4,
    "save_dir": "./save/train/baseline",
    "seed": 224,
 
    "train_eval_file": "./data/train_eval.json",
    "train_record_file": "./data/train.npz",
   
    "word_emb_file": "./data/word_emb.json"
}

class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'





class GEOROC(data.Dataset):

    def __init__(self, data_path):
        print(data_path)
        super(GEOROC, self).__init__()

        dataset = np.load(data_path)

        self.abstract_idxs = torch.from_numpy(dataset['abstract_idxs']).long()
#         self.abstract_char_idxs = torch.from_numpy(dataset['abstract_char_idxs']).long()
        self.title_idxs = torch.from_numpy(dataset['title_idxs']).long()
        self.y = torch.from_numpy(dataset['y']).long()
        self.citation_idxs = torch.from_numpy(dataset['citation_idxs']).long()
#         self.title_char_idxs = torch.from_numpy(dataset['title_char_idxs']).long()
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))]


    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.abstract_idxs[idx],
#                    self.abstract_char_idxs[idx],
                   self.title_idxs[idx],
                 self.citation_idxs[idx],
                 self.y[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.valid_idxs)


def collate_fn(examples):
    """
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    abstract_idxs, \
        title_idxs, \
        citation_idxs, \
        y,\
         ids = zip(*examples)

    # Merge into batch tensors
    abstract_idxs = merge_1d(abstract_idxs)
#     abstract_char_idxs = merge_2d(abstract_char_idxs)

    title_idxs = merge_1d(title_idxs)
#     title_char_idxs = merge_2d(title_char_idxs)
    ids = merge_0d(ids)
    citation_idxs = merge_0d(citation_idxs)


    return (abstract_idxs,
            title_idxs,
            citation_idxs,
            y,
             ids)






def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def encoder_fun(mineral,mineral_set):
    encoded_mineral=torch.zeros(1,len(mineral_set))

    mineral_set_L=list(mineral_set)
    mineral=mineral.split()
    for m in mineral:
        i=mineral_set_L.index(m)
        encoded_mineral[0][i]=1
    return encoded_mineral

def get_class(mineral_set,mineral_set_L,DATA,citation):
    has_cit=DATA.CITATIONS.eq(str(citation))
    if len(DATA[has_cit]) ==0:

        # print("length zero")
        return "notenough"
    y=DATA[has_cit]
    u1=y["MINERAL"].unique()
    tmp=''
    for u in u1:
        tmp=tmp+str(u)+' '

    y=y.iloc[0]#.landorsea_ENCODED

    y=y.MINERAL
    # tmp=y
    y=tmp
    y=str(y).lower()
    y=re.sub(";","",y)
    y=clean_text(y)
    y=y.replace("/"," ")
    y=y.replace("-"," ")
    y=y.replace("_"," ")
    y=encoder_fun(y,mineral_set)
    y=y[:,mineral_set_L.index("cpx")]
    y=y.to(torch.long)  
  
    return y.item()

def process_file(filename, data_type, word_counter, char_counter):
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
    # DATA[["MINERAL"]]

    mineral_set=set()
    DATA=DATA
    for r in DATA["MINERAL"].unique():
     
        for m in str(r).split():
            mineral_set.add(str(m).lower())
 
    mineral_set_L=list(mineral_set)
    meta_data=DATA
    data=DATA
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    i=0
    data=pd.read_csv(filename)


    for index, row in data.iterrows():
        i=i+1
        if i==100:
            break
        
        abstract_tokens_L=[]
        abstract_chars_L=[]
        abstract_tmp=[]
        citation = row["CITATIONS"]
        y=get_class(mineral_set,mineral_set_L,DATA,citation)
        if y=="notenough":
            print(y)
            continue

        abstract_t=[]

        abstracts=row["ABSTRACT"] #article["ABSTRACT"]
        abstracts=abstracts.replace(
            "_", ' ')
        abstract_tokens = word_tokenize(abstracts)
        abstract_chars = [list(token) for token in abstract_tokens]
  
        for token in abstract_tokens:
                word_counter[token] +=1
                for char in token:
                        char_counter[char] += 1

        title=row["TITLE"]  #article["TITLE"]
      
        title=title.replace(
            "_", ' ')
        title_tokens = word_tokenize(title)
        title_chars = [list(token) for token in title_tokens]
  
        for token in title_tokens:
                word_counter[token] +=1
                for char in token:
                        char_counter[char] += 1
           

        citation=row["CITATIONS"]  # article["CITATIONS"]

        if len(title_tokens)>40:
            title_tokens=title_tokens[:40]

        example = {"abstract_tokens": abstract_tokens,     
                   "abstract_chars": abstract_chars,          
                   "title_tokens": title_tokens,
                   "title_chars": title_chars,
                   "citation": citation,
                   "y":y,
                   "id": total}

        examples.append(example)

        eval_examples[str(total)] = {"supports": abstracts,
                                     "title":row["TITLE"],
                                     "citation": citation,
                               
                                     }
        total=total+1

        print(f"{len(examples)} abstracts in total")
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    # adapted from "https://github.com/minggg/squad/blob/master/setup.py"
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict



def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    # adapted from "https://github.com/minggg/squad/blob/master/setup.py"
    abstract_limit = 2000#args.test_abstract_limit if is_test else args.abstract_limit
    print("args.test_para_limit if is_test else args.para_limit")
    print(args.test_abstract_limit if is_test else args.abstract_limit)
    title_limit = 200
    # ans_limit = 2
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:

            drop = len(ex["abstract_tokens"]) > abstract_limit or \
                   len(ex["title_tokens"]) > title_limit 
              

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    abstract_idxs = []
    title_idxs = []
    abstract_char_idxs = []
    title_char_idxs = []
    CITATION=[]
    Y=[]
 

    ids = []
    for n, example in tqdm(enumerate(examples)):
      

        citation=example["citation"]
        CITATION.append(citation)
        total_ += 1

        if drop_example(example, is_test):
            continue
        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        abstract_idx = np.zeros([abstract_limit+150], dtype=np.int32)

        abstract_char_idx = np.zeros([abstract_limit, char_limit], dtype=np.int32)
      

        title_idx = np.zeros([40], dtype=np.int32)
        title_char_idx = np.zeros([abstract_limit, char_limit], dtype=np.int32) 

        for i, token in enumerate(example["abstract_tokens"]):
            abstract_idx[i] = _get_word(token)
        abstract_idxs.append(abstract_idx)   

        for i, token in enumerate(example["title_tokens"]):
            title_idx[i] = _get_word(token)
        title_idxs.append(title_idx)
        for i, token in enumerate(example["abstract_tokens"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                abstract_char_idx[i, j] = _get_char(char)
                # abstract_char_idxs.append(abstract_char_idx)

        for i, token in enumerate(example["title_tokens"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                title_char_idx[i, j] = _get_char(char)
                # title_char_idxs.append(title_char_idx)

        ids.append(example["id"])
        Y.append(example["y"])
    np.savez(out_file,
             abstract_idxs=np.array(abstract_idxs),
             # abstract_char_idxs=np.array(abstract_char_idxs),
 
             title_idxs=np.array(title_idxs),
             # title_char_idxs=np.array(title_char_idxs),

             citation_idxs=np.array(CITATION),

             y=np.array(Y),
             ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])




def save_or_read_data(tmp=False):

    data=pd.read_csv("data/data_final.csv")

    if tmp!=False:
        data.to_json(r'data/data_abstracts.json', orient='records')
    return data

def pre_process():
    # # adapted from "https://github.com/minggg/squad/blob/master/setup.py"

    args_ = get_setup_args()
    # print("args",args_)

    # # Download resources
    # download(args_)

    # # Import spacy language model
    nlp = spacy.blank("en")

    # print(args_.train_url)
    # exit()

    # Preprocess dataset
    args_.train_file ="data/data_train.csv"
    args_.dev_file ="data/data_eval.csv"

    args_.include_test_examples=False
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    args=args_




    train_file= args_.train_file
    dev_file=args_.dev_file 
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(train_file, "train", word_counter, char_counter)
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    # Process dev and test sets
    dev_examples, dev_eval = process_file(dev_file, "dev", word_counter, char_counter)
   

    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")

