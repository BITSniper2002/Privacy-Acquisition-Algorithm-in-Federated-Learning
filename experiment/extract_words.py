import csv
import os
import pickle

# Set the HF_HOME environment variable to a directory where you have write permissions
os.environ["HF_HOME"] = "./tmp"
import json
import torch
from dlg_utils import calculatePerplexity,calculateGradLoss,get_grad_gpt2
import copy
import random
import itertools
from itertools import permutations
import numpy as np
import sys

def createModels():
    import GPUtil
    from transformers import AutoModelForCausalLM, GPT2Tokenizer,AutoTokenizer

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("satkinson/DialoGPT-small-marvin")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = AutoModelForCausalLM.from_pretrained("satkinson/DialoGPT-small-marvin")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    gpt2 = gpt2.to(device)
    gpt2.eval()

    return tokenizer, gpt2

import torch
from torch.autograd import grad

def get_word_embedding_gradients(sentence, tokenizer_GPT, gpt2):
    tokens = tokenizer_GPT(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=gpt2.config.max_length,
    )["input_ids"]

    if not torch.is_tensor(tokens):
        tokens = torch.tensor(tokens).long()
    tokens = tokens.to(gpt2.device)

    gpt2.zero_grad()
    gpt2_outputs = gpt2(
        input_ids=tokens,
        output_hidden_states=True,
        return_dict=True,
        labels=tokens,
    )

    # Calculate loss
    loss = gpt2_outputs.loss

    # Compute gradients of the loss with respect to the word embeddings
    word_embedding_gradients = grad(
        loss,
        gpt2.transformer.wte.weight,  # Accessing word embedding weights
        retain_graph=True  # Retain graph for multiple calls to grad
    )[0]

    return word_embedding_gradients

#step 1: get bag-of-words from gradient observed
def tokens_in_sentence(embedding_gradients, threshold=9e-2):#make a figure for performance under different threshold
    # 判断每个token是否在句子中
    tokens_in_sentence = [torch.mean(embedding_gradient.abs()) > threshold for embedding_gradient in embedding_gradients]

    return tokens_in_sentence

def get_words(vocab,tokens_presence):
    token_ids = [token_id for token_id in range(len(tokens_presence)) if tokens_presence[token_id] == torch.tensor([True])]
    words = [vocab[ids] for ids in token_ids]
    return words







if __name__ == "__main__":
    # 加载tokenizer
    tokenizer, model = createModels()
    vocab_path = './DialoGPT-small-marvin/vocab.json'

    with open('./demonstration/demon.json', 'r') as file:
        data = json.load(file)
    sentences = data

    # 读取 JSON 文件中的字典
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    vocab = {vocab[i]: i for i in vocab}
    #获取每个句子的梯度。梯度是大小为（50257，768）的张量
    gradients = [get_word_embedding_gradients(sentence=sentence, tokenizer_GPT=tokenizer, gpt2=model) for sentence in
                 sentences]
    words = []
    total_words = 0
    #bag-of-words extraction
    for gradient in gradients:
        tokens_presence = tokens_in_sentence(gradient)
        cnt = 0
        for token in tokens_presence:
            if token == True:
                cnt += 1
        print(cnt)
        total_words += cnt
        curr_words = get_words(vocab,tokens_presence)
        for i in range(len(curr_words)):
            curr_words[i] = str(curr_words[i].replace('Ġ',''))
        print(curr_words)
        words.append(curr_words)
    csv_file = "./demonstration/demon.csv"
    # csv_file = "wiki.csv"
    # 打开文件并写入列表内容
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(words)


