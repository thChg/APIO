import logging
import nltk
import re
import Levenshtein as lev
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from transformers import XLNetTokenizer, XLNetModel

def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='{asctime} - {levelname} - {message}',
        datefmt='%Y-%m-%d %H:%M',
        style='{'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def tokenize_text(text: str):
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def detokenize_text(text: str):
    tokens = word_tokenize(text)
    return TreebankWordDetokenizer().detokenize(tokens)

def levenstein_distance(text1, text2):
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)
    return lev.distance(tokens1, tokens2)


def extract_tagged_text(text: str, begin_tag: str, end_tag: str) -> str:
    pattern = re.escape(begin_tag) + r'(.*?)' + re.escape(end_tag)
    match = re.search(pattern, text, re.DOTALL)  # Use re.DOTALL to allow newlines

    if match:
        return match.group(1).strip().replace('<', '').replace('>', '').strip()

    return ''

def embedding(text):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state 
        sentence_embedding = torch.mean(last_hidden_states, dim=1)  
    
    return sentence_embedding.squeeze().numpy()

def semantic_distance(text1, text2):
    embedding1 = embedding(text1).reshape(1, -1)
    embedding2 = embedding(text2).reshape(1, -1)
    return 1 - cosine_similarity(embedding1, embedding2)[0][0]