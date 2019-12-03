from encoders import NMTEncoder
from decoders import NMTDecoder
from data_loader import NMTDataset, generate_nmt_batches
from models import NMTModel
import sampler
import utils_inference
import os
from argparse import Namespace
from collections import Counter
import json
import re
import string
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
from math import log
from nltk.translate import bleu_score
import seaborn as sns
import matplotlib.pyplot as plt
from alignment_utils import tokenize_mr, tokenize_mr_upper
from data_processing import Delexicalizer
from slot_aligner import SlotAligner
from alignment_utils import tokenize_mr, tokenize_mr_upper
import sampler

args = Namespace(dataset_csv="data/inp_and_gt.csv",
                 vectorizer_file="vectorizer.json",
                 model_state_file="model.pth",
                 save_dir="data/model_storage/",
                 cuda=True,
                 seed=1337,
                 batch_size=32)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.cuda.manual_seed_all(args.seed)

args.device = torch.device("cuda" if args.cuda else "cpu")

dataset = NMTDataset.load_dataset_and_load_vectorizer(args.dataset, args.vectorizer_file)

vectorizer = dataset.get_vectorizer()

model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)

model.load_state_dict(torch.load(args.save_dir+args.model_state_file))
model.eval().to(args.device)

inference_sampler = sampler.NMTSampler(vectorizer, model, use_reranker=True, beam_width=3)
dataset.set_split('val')

batch_generator = generate_nmt_batches(dataset, 
                                       batch_size=args.batch_size, 
                                       device=args.device,
                                       shuffle=False)

#Contains the samples for the current batch
batch_dict = next(batch_generator)

sampler.apply_to_batch(batch_dict)

all_results = []

for i in range(args.batch_size):
    all_results.append(sampler.get_ith_item(i, False))

for result in all_results:
    print("MR GT: ")
    print(result["mrs_gt"])
    print("------------------")
    print("Input MR: ")
    print(result["source"])
    print("-------------------------")
    print("Reference ground truth: ")
    print(result["reference_gt"])
    print("---------------------------")
    print("Sampled no delexicalization: ")
    print(result["sampled_normalized"])
    print("---------------------------")
    print("Sampled with delexicalization: ")
    print(" ".join(result["sampled"]))
    print("---------------------------")
    print("############################")



