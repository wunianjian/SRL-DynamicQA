import os
import copy
import json
import nltk
import torch
import spacy
import shutil
import random
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def batch_by_tokens(length, max_tokens):
    indexes = []
    i = 0
    while i < len(length):
        for j in range(i, len(length)):
            maxc = max(length[i:j + 1])
            maxn = max(maxc, length[min(j + 1, len(length) - 1)])
            current_batch_tokens = maxc * (j + 1 - i)
            next_batch_tokens = maxn * (j + 2 - i)
            if (current_batch_tokens <= max_tokens < next_batch_tokens) or j == len(length) - 1:
                indexes.append((i, j))
                i = j + 1
                break
    return indexes


ARGS_DESC = {
    '0': 'agent',
    '1': 'patient',
    '2': 'instrument, benefactive or attribute',
    '3': 'starting point, benefactive or attribute',
    '4': 'ending point',
    '5': 'direction, attribute or instrument',
    'A': 'causal agent'
}

ARGMS_DESC = {
    'MNR': 'manner',
    'ADV': 'adverbials',
    'LOC': 'locative',
    'TMP': 'temporal',
    'PRP': 'purpose clauses',
    'PRD': 'secondary predication',
    'DIR': 'directional',
    'DIS': 'discourse',
    'MOD': 'modal',
    'NEG': 'negation',
    'CAU': 'cause clauses',
    'EXT': 'extent',
    'LVB': 'light verb',
    'REC': 'reciprocals',
    'ADJ': 'adjectival',
    'GOL': 'goal',
    'DSP': 'direct speech',
    'PRR': 'predicating relation',
    'COM': 'comitative',
    'PRX': 'predicating expression',
    'PNC': 'purpose not cause'
}

ARGS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA']
ARGMS = None

TAGS = ['O', 'B-N', 'B-C', 'B-R', 'I-N', 'I-C', 'I-R']
TAGS2ID = dict([(j, i) for i, j in enumerate(TAGS)])

LABELS = None
LABELS1 = None
ALL_LABELS = None

frames = json.load(open('./data/conll2012/frames.json'))
multi_arg_num = 0

en_nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')
nltk.download('omw-1.4')


def get_root(sen):
    return next(en_nlp(sen).sents).root.text


def get_noun_root(sen):
    tokens_no_v = []
    doc = en_nlp(sen)
    if len(doc) == 1:
        return sen
    for token in doc:
        if not str(token.tag_).startswith('V'):
            tokens_no_v.append(token.text)
    if len(tokens_no_v) == 0:
        return ''
    return get_root(' '.join(tokens_no_v))


def get_all_pos(word):
    pos_l = set()
    for tmp in wn.synsets(word):
        if tmp.name().split('.')[0] == word:
            pos_l.add(tmp.pos())
    return pos_l


def is_person(desc):
    sens = desc.split(',')
    for sen in sens:
        n_root = get_noun_root(sen)
        if n_root.endswith('er') or n_root.endswith('or'):
            lemma = n_root[:-2]
            if 'v' in get_all_pos(lemma) or 'v' in get_all_pos(lemma + 'e'):
                return True
    return False


def get_desc(frame_key, label2):
    try:
        desc = frames[frame_key]['args'][label2]
    except KeyError:
        desc = ARGS_DESC[label2]
    return desc


def get_query(desc, core_info):
    if is_person(desc):
        return f'who is the {desc} {core_info}?'
    else:
        return f'what is the {desc} {core_info}?'


def label2query_naive(label, frame_key, core_info, arg_dict):
    label_parts = label.split('-')
    if label_parts[0] in ('R', 'C'):
        if label_parts[0] == 'R':
            question = 'what is the reference of '
        else:
            question = 'what is the continuation of '
        try:
            query = question + ' '.join(arg_dict['-'.join(label_parts[1:])][-1]) + '?'
        except KeyError:
            if len(label_parts) == 2:
                query = get_query(get_desc(frame_key, label[-1]), '')
            else:
                query = get_query(ARGMS_DESC[label_parts[-1]], core_info)
    else:
        if len(label_parts) == 1:
            assert len(label) == 4, (label, frame_key)
            label2 = label[-1]
            desc = get_desc(frame_key, label2)
        else:
            desc = ARGMS_DESC[label_parts[-1]]
        query = f'what is the argument of predicate {frame_key.split(".")[0]} with meaning {desc} ?'
    return query


def label2query(label, frame_key, core_info, arg_dict):
    """
    Args:
        label: arg label
        frame_key: frame key
        core_info: known answer of other roles
        arg_dict: argument dict
    """
    label_parts = label.split('-')
    if label_parts[0] in ('R', 'C'):
        if label_parts[0] == 'R':
            question = 'what is the reference of '
        else:
            question = 'what is the continuation of '
        try:
            query = question + ' '.join(arg_dict['-'.join(label_parts[1:])][-1]) + '?'
        except KeyError:
            if len(label_parts) == 2:
                query = get_query(get_desc(frame_key, label[-1]), '')
            else:
                query = get_query(ARGMS_DESC[label_parts[-1]], core_info)
    else:
        if len(label_parts) == 1:
            assert len(label) == 4, (label, frame_key)
            label2 = label[-1]
            desc = get_desc(frame_key, label2)
        else:
            desc = ARGMS_DESC[label_parts[-1]]
        query = get_query(desc, core_info.strip())
    return query


def get_samples_for_sen(sen_id, sen_dict, gold_level):
    sentence = sen_dict['sentence']
    predicates = sen_dict['predicates']
    arguments_list = sen_dict['arguments']
    p_labels_list = sen_dict['plabel']
    if gold_level == 0:
        lemmas = sen_dict['lemmas']
        frameset_ids = sen_dict['frameset_ids']
    else:
        p_lemma_ids = sen_dict['plemma_ids']
        lemmas, frameset_ids = [], []
        if len(p_lemma_ids) > 0:
            lemmas, frameset_ids = zip(*[x.split('.') for x in p_lemma_ids])
    context = ''
    id2pos = {}
    for i, word in enumerate(sentence):
        id2pos[i] = (len(context), len(context) + len(word))
        context += word + ' '
    context = context.strip()
    samples = []
    golds = []
    ids = []
    for i in range(len(predicates)):
        pred_id = predicates[i]
        p_label = p_labels_list[i]
        arguments = arguments_list[i]
        frameset_id = frameset_ids[i]
        lemma = lemmas[i]
        frame_key = '.'.join([lemma, frameset_id])
        arg_dict = {}
        for (s_id, e_id, arg_label) in arguments:
            if arg_label not in arg_dict:
                arg_dict[arg_label] = ([], [], [])
            arg_dict[arg_label][0].append(id2pos[s_id][0])
            arg_dict[arg_label][1].append(id2pos[e_id][1])
            arg_dict[arg_label][2].append(context[id2pos[s_id][0]: id2pos[e_id][1]])
            if arg_label != 'V':
                golds.append((sen_id, i, arg_label, id2pos[s_id][0], id2pos[e_id][1]))
            assert context[id2pos[s_id][0]: id2pos[e_id][1]] == ' '.join(sentence[s_id: e_id + 1]), \
                (context[id2pos[s_id][0]: id2pos[e_id][1]], ' '.join(sentence[s_id: e_id + 1]))
        assert arg_dict['V'][0][0] == id2pos[pred_id][0], f"V is not predicate: {i}, {sen_dict}"
        arg_sort_list = sorted(arg_dict.items(), key=lambda x: x[1][0][0])
        core_args = [(s_pos, arg_text) for arg_label, (s_poses, _, arg_texts) in arg_sort_list if
                     arg_label in ('V', 'ARG0', 'ARG1') for s_pos, arg_text in zip(s_poses, arg_texts)]
        sorted_core_args = sorted(core_args, key=lambda x: x[0])
        core_info = ' '.join(list(zip(*sorted_core_args))[1])
        del arg_dict['V']
        excluded_p_label = []
        for p_arg in p_label:
            for prefix in ('', 'R-', 'C-'):
                if len(p_arg) == 3:
                    prefix += 'ARGM-'
                p_arg_l = prefix + p_arg
                if p_arg_l not in arg_dict:
                    excluded_p_label.append(p_arg_l)
        # balance the number of negative samples while training
        if len(excluded_p_label) > len(arg_dict) // 4:
            excluded_p_label = random.sample(excluded_p_label, len(arg_dict) // 4)
        for p_arg_l in excluded_p_label:
            arg_dict[p_arg_l] = ([0], [0], [''])
        for arg_label, (s_poses, e_poses, arg_texts) in arg_dict.items():
            if arg_label == 'V':
                continue
            if arg_label in ('ARG0', 'ARG1'):
                question = label2query(arg_label, frame_key, '', arg_dict)
            else:
                question = label2query(arg_label, frame_key, core_info, arg_dict)
            s_poses, e_poses, arg_texts = zip(*sorted(zip(s_poses, e_poses, arg_texts), key=lambda x: -len(x[-1])))
            sample = {
                'answers': {'answer_start': s_poses, 'text': arg_texts},
                'context': context,
                'question': question,
            }
            samples.append(sample)
            ids.append((sen_id, i, arg_label))
    return samples, golds, ids


class MyDataset(Dataset):
    def __init__(self, path="", tokenizer=None, max_tokens=1024, gold_level=0, device='cuda:0'):
        """
        gold_level=0: gold predicate disambiguation
        gold_level=1: predict predicate disambiguation

        arg_query_type=0: query with label only
        arg_query_type=1: query without label but with semantics
        arg_query_type=2: query with label and semantics
 
        argm_query_type=0: query with label only
        argm_query_type=1: query with semantics
        """
        # get initial features
        if not path:
            return
        data = json.load(open(path))
        self.sample_list = []
        self.data = copy.deepcopy(data)
        self.tokenized_data = []
        self.input_ids = []
        self.token_type_ids = []
        self.target = []
        self.ids = []
        self.gold = []
        # the evaluation for CONL 2009 includes the results of predicate disambiguation
        self.gold_senses = []
        self.senses = []  # predict sense under predict lemma
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device
        self.data_dir = os.path.dirname(path)
        if 'train' in path:
            self.dataset = 'train'
        else:
            self.dataset = 'dev'
        if os.path.exists(os.path.join(self.data_dir, f'{self.dataset}.samples.json')):
            with open(os.path.join(self.data_dir, f'{self.dataset}.samples.json'), 'r') as fp:
                self.sample_list = json.load(fp)
            with open(os.path.join(self.data_dir, f'{self.dataset}.golds.json'), 'r') as fp:
                self.gold = json.load(fp)
            with open(os.path.join(self.data_dir, f'{self.dataset}.ids.json'), 'r') as fp:
                self.ids = json.load(fp)
        else:
            self.init_data(self.data, gold_level)

    def init_data(self, data, gold_level):
        for s_id, d in enumerate(tqdm(data, desc='preprocessing')):
            samples, golds, ids = get_samples_for_sen(s_id, d, gold_level)
            self.sample_list.extend(samples)
            self.gold.extend(golds)
            self.ids.extend(ids)
        with open(os.path.join(self.data_dir, f'{self.dataset}.samples.json'), 'w') as fp:
            json.dump(self.sample_list, fp)
        with open(os.path.join(self.data_dir, f'{self.dataset}.golds.json'), 'w') as fp:
            json.dump(self.gold, fp)
        with open(os.path.join(self.data_dir, f'{self.dataset}.ids.json'), 'w') as fp:
            json.dump(self.ids, fp)

    def preprocess_function(self, example):
        inputs = self.tokenizer(
            example["question"].strip(),
            example["context"],
            max_length=self.max_tokens,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset = inputs.pop("offset_mapping")
        answer = example["answers"]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])

        if start_char == end_char == 0:
            inputs["start_positions"] = 0
            inputs["end_positions"] = 0
        else:
            context_start = inputs['input_ids'].index(2) + 2
            for i, (s, e) in enumerate(offset[context_start:]):
                if s == start_char:
                    inputs["start_positions"] = i
                if e == end_char:
                    inputs["end_positions"] = i
                    break
        return inputs

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, i):
        batch = self.preprocess_function(self.sample_list[i])
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch


def load_data(path, pretrained_model_name_or_path, max_tokens, shuffle, gold_level=0, device='cuda:0'):
    global LABELS1, ALL_LABELS, ARGMS
    ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
             'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']
    ALL_LABELS = ['O'] + [t1 + '-' + t0 for t0 in ARGS + ARGMS for t1 in TAGS[1:]]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    # tokenizer.add_special_tokens(
    #     {'additional_special_tokens': ['<p>', '</p>']})
    dataset = MyDataset(path, tokenizer, max_tokens,
                        gold_level, device)
    return dataset
