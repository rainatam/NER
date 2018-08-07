import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    TIT = get_name_entitry('TIT', tag_seq, char_seq)
    JOB = get_name_entitry('JOB', tag_seq, char_seq)
    DOM = get_name_entitry('DOM', tag_seq, char_seq)
    EDU = get_name_entitry('EDU', tag_seq, char_seq)
    WRK = get_name_entitry('WRK', tag_seq, char_seq)
    SOC = get_name_entitry('SOC', tag_seq, char_seq)
    AWD = get_name_entitry('AWD', tag_seq, char_seq)
    PAT = get_name_entitry('PAT', tag_seq, char_seq)
    PRJ = get_name_entitry('PRJ', tag_seq, char_seq)
    return TIT, JOB, DOM, EDU, WRK, SOC, AWD, PAT, PRJ

def get_name_entitry(name, tag_seq, char_seq):
    length = len(char_seq)
    lst = []
    item = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-'+name:
            if len(item) > 0:
                lst.append(item)
            item = char
            if i+1 == length:
                lst.append(item)
        if tag == 'I-'+name:
            item += char
            if i+1 == length:
                lst.append(item)
        if tag not in ['I-'+name, 'B-'+name]:
            if len(item) > 0:
                lst.append(item)
                item = ''
            continue
    return lst



def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
