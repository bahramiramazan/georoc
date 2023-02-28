""" # adapted from "https://github.com/minggg/squad/blob/master/args.py"

"""

import argparse


def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Georoc')

    add_common_args(parser)
    parser.add_argument('--task',
                        type=str,
                        default='preprocess')

 
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='./data/dev_meta.json')

    parser.add_argument('--word2idx_file',
                        type=str,
                        default='./data/word2idx.json')
    parser.add_argument('--char2idx_file',
                        type=str,
                        default='./data/char2idx.json')
  
    parser.add_argument('--abstract_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a abstract')
    parser.add_argument('--title_limit',
                        type=int,
                        default=50,
                        help='Max number of words to keep from a title')
    parser.add_argument('--test_abstract_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a abstract at test time')
    parser.add_argument('--test_title_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a title at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')
  
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')


    args = parser.parse_args()

    return args




def add_common_args(parser):

    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
  
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='./data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
  
