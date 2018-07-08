#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os, sys, argparse, time, random

sys.path.append('.')

from util import *
from model import BiLSTM_CRF, Original_model
from utils import str2bool, get_logger, get_entity, get_name_entitry
from data import read_corpus, read_dictionary, random_embedding


DATA_DIR = os.path.join(os.path.abspath('..'), 'data')

MODEL3_PATH = "../model/data_path_save/1521112368/checkpoints/"
# MODEL3_PATH = "../model/data_path_save/1530423394/checkpoints/"
# MODEL_PATH = "../model/data_path_save/1530521907/checkpoints/" #5
# MODEL_PATH = "../model/data_path_save/1530605248/checkpoints/" #12
# MODEL_PATH = "../model/data_path_save/1530683206/checkpoints/" #7
MODEL_PATH = "../model/data_path_save/1530721857/checkpoints/" #9

NER_PATH = '.'

#=============== SET UP =================================================================================

config = tf.ConfigProto()

parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()

paths = {}
timestamp = '1521112368'
output_path = os.path.join(NER_PATH, "data_path_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))
paths['restore_path'] = ''

word2id = read_dictionary(os.path.join(NER_PATH, args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

#========================================================================================================

with open(os.path.join(DATA_DIR, 'text.txt')) as file:
    data = file.read()
personList = data.split('*********&&&&&&&&')

def print_tag(lst, name, text):
    temp = clean_list(lst)
    text = clean_word(text)
    cnt =  [text.count(word) for word in temp]
    print(name, ': ', end='')
    for i, v in enumerate(temp):
        print(str(temp[i])+'('+str(cnt[i])+'),', end=' ')
    print('')

def extract_one(text):
    text = clean_text(text).strip()
    if len(text) == 0:
        return

    tag2label = {"O": 0,
             "B-TIT": 1, "I-TIT": 2,
             "B-JOB": 3, "I-JOB": 4,
             "B-DOM": 5, "I-DOM": 6,
             "B-EDU": 7, "I-EDU": 8,
             "B-WRK": 9, "I-WRK": 10,
             "B-SOC": 11, "I-SOC": 12,
             "B-AWD": 13, "I-AWD": 14,
             "B-PAT": 15, "I-PAT": 16,
             "B-PRJ": 17, "I-PRJ": 18 
             }
    ckpt_file = tf.train.latest_checkpoint(MODEL_PATH)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt_file)
        
        demo_sent = list(text)
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo(sess, demo_data, tag2label)

        TIT = get_name_entitry('TIT', tag, demo_sent)
        JOB = get_name_entitry('JOB', tag, demo_sent)
        DOM = get_name_entitry('DOM', tag, demo_sent)
        EDU = get_name_entitry('EDU', tag, demo_sent)
        WRK = get_name_entitry('WRK', tag, demo_sent)
        SOC = get_name_entitry('SOC', tag, demo_sent)
        AWD = get_name_entitry('AWD', tag, demo_sent)
        PAT = get_name_entitry('PAT', tag, demo_sent)
        PRJ = get_name_entitry('PRJ', tag, demo_sent)
    sess.close()
    return TIT, JOB, DOM, EDU, WRK, SOC, AWD, PAT, PRJ

def extract_one_3(text):
    text = clean_text(text).strip()
    if len(text) == 0:
        return
    tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-ADR": 3, "I-ADR": 4,
             "B-AFF": 5, "I-AFF": 6,
             }
    ckpt_file = tf.train.latest_checkpoint(MODEL3_PATH)
    paths['model_path'] = ckpt_file
    model = Original_model(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    # sess2 = tf.Session(config=config)
    # with sess2.as_default():
    with tf.Session(config=config) as sess2:
        tf.get_variable_scope().reuse_variables()
        saver.restore(sess2, ckpt_file)
        
        demo_sent = list(text)
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo(sess2, demo_data, tag2label)

        PER = get_name_entitry('PER', tag, demo_sent)
        ADR = get_name_entitry('ADR', tag, demo_sent)
        AFF = get_name_entitry('AFF', tag, demo_sent)
        return clean_list(PER), clean_list(ADR), clean_list(AFF)

def interface(text):
    TIT, JOB, DOM, EDU, WRK, SOC, AWD, PAT, PRJ = extract_one(text)
    tf.reset_default_graph()
    PER, ADR, AFF = extract_one_3(text)
    print_tag(PER, 'PER', text)
    print_tag(ADR, 'ADR', text)
    print_tag(AFF, 'AFF', text)
    print_tag(TIT, 'TIT', text)
    print_tag(JOB, 'JOB', text)
    print_tag(DOM, 'DOM', text)
    print_tag(EDU, 'EDU', text)
    print_tag(WRK, 'WRK', text)
    print_tag(SOC, 'SOC', text)
    print_tag(AWD, 'AWD', text)
    print_tag(PAT, 'PAT', text)
    print_tag(PRJ, 'PRJ', text)
    
    return PER, ADR, AFF, TIT, JOB, DOM, EDU, WRK, SOC, AWD, PAT, PRJ
    
interface(personList[1])
