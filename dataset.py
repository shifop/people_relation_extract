"""
实现数据加载操作
"""
import json
from transformers import BertTokenizer
from numpy.core.fromnumeric import sort
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import random
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
import os
from config import cfg
random.seed(0)
tf.random.set_seed(0)


class Dataset(object):
    def __init__(self, process_data=None, batch_size=8):
        self.process = process_data
        self.tokenizer = BertTokenizer.from_pretrained(cfg.DIR)
        self.total = len(self.process)//batch_size
        self.num = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        sep_indexes = []
        labels = []
        input_ids = []
        attention_mask = []
        token_type_ids = []

        data = self.process[self.num*self.batch_size:(self.num+1)*self.batch_size]
        if len(data)!=self.batch_size:
            self.num = 0
            raise StopIteration

        for _ in data:
            keyindex, tag, content = _
            labels.append(tag)
            sep_indexes.append([i+1 for i in keyindex])
            content = self.tokenizer.encode_plus(
                    content,
                    max_length=cfg.MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    pad_to_multiple_of=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='tf'
                    )

            input_ids.extend(content['input_ids'].numpy().tolist())
            attention_mask.extend(content['attention_mask'].numpy().tolist())
            token_type_ids.extend(content['token_type_ids'].numpy().tolist())

        self.num+=1
        return input_ids, token_type_ids, attention_mask, sep_indexes, labels


def get_dataset_by_iter(trainset):
    def get_dataset():
        for input_ids, token_type_ids, attention_mask, sep_index, labels in trainset:
            yield input_ids, token_type_ids, attention_mask, sep_index, labels

    return get_dataset


def get_data(fn, max_length=cfg.MAX_LENGTH,repeat=1, prefetch=64, shuffle=64, seed=0, count=0):
    dataset = tf.data.Dataset.from_generator(fn,
                                             (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                                             (tf.TensorShape([None, max_length]), tf.TensorShape([None, max_length]),
                                              tf.TensorShape([None, max_length]), tf.TensorShape([None, 4]), tf.TensorShape([None]))
                                             )

    if shuffle!=0:
        dataset = dataset.repeat(count=repeat).prefetch(prefetch).shuffle(shuffle,seed=seed)
    else:
        dataset = dataset.repeat(count=repeat).prefetch(prefetch)

    return dataset


if __name__=='__main__':
    def read_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

    data = read_json('./data/train_p.json')
    data = Dataset(data)
    for _ in data:
        print('')