import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import *
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
from accelerate import Accelerator

# 全局变量
DATASET = 'Sports'
ROOT = "/kaggle/working/MQL4GRec/datasets/LC-Rec_image"
GPU_IDS = [0, 1]  # 使用两个GPU
PLM_NAME = 'llama'
MODEL_NAME_OR_PATH = 'huggyllama/llama-7b'
MODEL_CACHE_DIR = '/kaggle/tmp/cache_model'
MAX_SENT_LEN = 2048
WORD_DROP_RATIO = -1  # 默认不丢弃单词
BATCH_SIZE = 2  # 进一步降低batch_size以减少显存需求

def load_data():
    item2feature_path = os.path.join(ROOT, f'{DATASET}.item.json')
    item2feature = load_json(item2feature_path)
    return item2feature

def generate_text(item2feature, features):
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())
        item_text_list.append([int(item), text])
    return item_text_list

def preprocess_text():
    print('Process text data: ')
    print(' Dataset: ', DATASET)
    item2feature = load_data()
    item_text_list = generate_text(item2feature, ['title', 'description'])
    return item_text_list

def generate_item_embedding(item_text_list, tokenizer, model, accelerator, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', DATASET)
    
    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start = 0
    with torch.no_grad():
        while start < len(order_texts):
            if (start + BATCH_SIZE) % 100 == 0:
                print(f"==> Processing batch {start + BATCH_SIZE}")
            field_texts = order_texts[start: start + BATCH_SIZE]
            field_texts = list(zip(*field_texts))
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                if word_drop_ratio > 0:
                    print(f'Word drop with p={word_drop_ratio}')
                    new_sentences = []
                    for sent in sentences:
                        new_sent = []
                        sent = sent.split(' ')
                        for wd in sent:
                            rd = random.random()
                            if rd > word_drop_ratio:
                                new_sent.append(wd)
                        new_sent = ' '.join(new_sent)
                        new_sentences.append(new_sent)
                    sentences = new_sentences
                encoded_sentences = tokenizer(sentences, max_length=MAX_SENT_LEN,
                                            truncation=True, return_tensors='pt', padding="longest")
                
                # 使用accelerator准备输入数据
                encoded_sentences = accelerator.prepare(encoded_sentences)
                
                # 打印显存使用情况
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        print(f"GPU memory allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB on cuda:{i}")
                
                outputs = model(input_ids=encoded_sentences.input_ids,
                              attention_mask=encoded_sentences.attention_mask)
    
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                field_embeddings.append(mean_output)
                
                # 释放显存
                del encoded_sentences, outputs, masked_output
                torch.cuda.empty_cache()
    
            field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
            embeddings.append(field_mean_embedding)
            start += BATCH_SIZE

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(ROOT, DATASET + '.emb-' + PLM_NAME + "-td" + ".npy")
    np.save(file, embeddings)

if __name__ == '__main__':
    ROOT = os.path.join(ROOT, DATASET)
    
    # 初始化Accelerator
    accelerator = Accelerator(mixed_precision="fp16")  # 使用混合精度以减少显存占用
    
    print(f"Using {accelerator.num_processes} GPUs!")
    
    item_text_list = preprocess_text()
    
    kwargs = {"cache_dir": MODEL_CACHE_DIR}
    plm_tokenizer, plm_model = load_plm(MODEL_NAME_OR_PATH, kwargs)
    
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    
    # 使用accelerator准备模型
    plm_model = accelerator.prepare(plm_model)
    
    generate_item_embedding(item_text_list, plm_tokenizer, plm_model, accelerator, word_drop_ratio=WORD_DROP_RATIO)