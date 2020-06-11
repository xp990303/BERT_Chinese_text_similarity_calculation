# -*- coding:UTF-8 -*-
import time
from string import punctuation
import codecs
import re
import jieba
import os
import numpy as np
import json
import gensim
from collections import Counter
from pytorch_pretrained_bert import BertTokenizer

PAD_WORD_INDEX = 0
OOV_WORD_INDEX = 1
MAX_WORD_LENGTH = 100
MAX_3GRAM_LENGTH = 200


def word_tokenize(text):
    """分词器"""
    return [token for token in jieba.cut(text)]


def split_sent(sent, qrepr, ngram_size=3):
    """将sent切分成tokens"""
    if qrepr == 'word':
        return [token for token in sent]
        # return word_tokenize(sent)
    elif qrepr.endswith("gram"):
        trigram = []
        # original_sent = word_tokenize(sent)
        original_sent = [token for token in sent]
        for i in range(len(original_sent) - 2):
            trigram.append(original_sent[i] + original_sent[i+1] + original_sent[i+2])
        return trigram
    else:
        raise Exception("Unrecognized represention %s !" % qrepr)


def read_sentences(path, vocab, is_train, repr='word', ngram_size=3, test_vocab=None):
    question = []
    max_len = 0
    punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
    with codecs.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = re.sub(r"[{}]+".format(punc), " ", line)
            #  去标点后的句子
            # print("line:", line)
            q_tokens = split_sent(line, repr, ngram_size)
            # 分词之后的句子
            # print("q_tokens:", q_tokens)
            token_ids = []
            if len(q_tokens) > max_len:
                max_len = len(q_tokens)
            for token in q_tokens:
                if token not in vocab[repr]:
                    if is_train:
                        vocab[repr][token] = len(vocab[repr])
                    elif repr == 'word' and token not in test_vocab[repr]:
                        test_vocab[repr][token] = len(vocab[repr]) + len(test_vocab[repr])
                # 上面是完成vocab和test_vocab词表，下面是将对应的id找出来填入token_ids
                if token in vocab[repr]:
                    token_ids.append(vocab[repr][token])
                elif repr == 'word':
                    token_ids.append(test_vocab[repr][token])
                else:
                    token_ids.append(OOV_WORD_INDEX)
            # 转为id后的句子
            # print("token_ids:", token_ids)
            question.append(token_ids)
    return question, max_len

def select_best_length2(path1, path2, limit_rate=0.95):
    """选择最佳的样本max_length"""
    len_list = []
    max_length = 0
    cover_rate = 0.0
    seq1 = []
    seq2 = []
    # 分别读a.toks和b.toks文件
    with codecs.open(path1, "r", encoding='utf-8') as f:
        with codecs.open(path2, "r", encoding='utf-8')as f2:
            for line1 in f:
                # print("q1", line1)
                seq1.append(line1)
            for line2 in f2:
                # print("q2", line2)
                seq2.append(line2)
    for i in range(len(seq1)):
        len_list.append(len(seq1[i]) + len(seq2[i]))
    number = len(seq1)
    all_sent = len(len_list)
    sum_length = 0
    len_dict = Counter(len_list).most_common()
    for i in len_dict:
        sum_length += i[0] * i[1]
    average_length = sum_length / all_sent
    for i in len_dict:
        rate = i[1] / all_sent
        cover_rate += rate
        if cover_rate >= limit_rate:
            max_length = i[0]
            break
    print("max_length: ", max_length)
    return max_length, number, seq1, seq2


def convert_data2(path1, path2, max_length, number, seq1, seq2):
    """转ID，进行padding，再加上CLP、SEP之后"""
    tokenizer = BertTokenizer('./model/bert-base-chinese/vocab.txt')
    input_id = []
    input_mask = []
    segment_id = []
    # number = 0
    print(len(seq1))

    for i in range(number):
        tokens_a = tokenizer.tokenize(seq1[i])
        tokens_b = tokenizer.tokenize(seq2[i])
        # print(seq2[i])
        # print(tokens_b)
        while True:
            if (len(tokens_a) + len(tokens_b)) <= max_length - 3:
                break
            else:
                # print(tokens_b)
                # tokens_b.pop()
                tokens_a = tokens_a[: int((max_length - 3) * len(tokens_a)/(len(tokens_a) + len(tokens_b)))]
                tokens_b = tokens_b[: int((max_length - 3) * len(tokens_b)/(len(tokens_a) + len(tokens_b)))]
        # 头尾加上[CLS] [SEP]标签
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens = tokens_a + tokens_b + ['[SEP]']
        input_id_ = tokenizer.convert_tokens_to_ids(tokens)
        segment_id_ = [0] * len(tokens_a) + [1] * (len(tokens_b) + 1)
        input_mask_ = [1] * len(tokens)
        # segment_id是用于区分token_a和token_b的
        # input_mask用于区分padding
        padding_ = [0] * (max_length - len(tokens))
        # 所有的输入进入bert的配置参数都要加上padding
        input_id_ += padding_
        segment_id_ += padding_
        input_mask_ += padding_
        # 每条语句放入列表中[sentence_num, MAX_LENGTH]
        input_id.append(input_id_)
        input_mask.append(input_mask_)
        segment_id.append(segment_id_)

    return input_id, input_mask, segment_id


def gen_data2(path, datasets):
    input_ids, input_masks, segment_ids = [], [], []
    all_sim_list = []
    for data_name in datasets:
        print(data_name)
        data_folder = "%s/%s" % (path, data_name)       # data/CHIPmedical/train-
        # 计算句子最大长度
        max_len, number, seq1, seq2 = select_best_length2("%s/a.toks" % data_folder, "%s/b.toks" % data_folder)
        print("creating datasets %s" % data_name)
        t = time.time()
        # 三个类型输入转换为id
        input_id, input_mask, segment_id = convert_data2("%s/a.toks" % data_folder, "%s/b.toks" % data_folder,
                                                         max_len + 3, number, seq1, seq2)
        # 标签转换为id
        sim_list = read_relevance("%s/sim.txt" % data_folder)

        input_ids.extend(input_id)
        input_masks.extend(input_mask)
        segment_ids.extend(segment_id)
        all_sim_list.extend(sim_list)
    # 打包数据
    data = {'sim': np.array(all_sim_list), 'input_id': np.array(input_ids), 'input_mask': np.array(input_masks),
            'segment_id': np.array(segment_ids)}

    return data


def read_relevance(path):
    """ 加载label文件"""
    sims = []
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f):
                sims.append(int(line.strip()))
    print("sims:", sims[0:5])
    return sims









