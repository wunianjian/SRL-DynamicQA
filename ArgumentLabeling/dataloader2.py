import os
import copy
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler


def batch_by_tokens(length, max_tokens):
    indexes = []
    i = 0
    while i < len(length):
        for j in range(i, len(length)):
            maxc = max(length[i:j+1])
            maxn = max(maxc, length[min(j+1, len(length)-1)])
            current_batch_tokens = maxc*(j+1-i)
            next_batch_tokens = maxn*(j+2-i)
            if (current_batch_tokens <= max_tokens and next_batch_tokens > max_tokens) or j == len(length)-1:
                indexes.append((i, j))
                i = j+1
                break
    return indexes


ARGS_DESC = {
    'A0': 0,
    'A1': 1,
    'A2': 2,
    'A3': 3,
    'A4': 4,
    'A5': 5,
    'AA': 'a',
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
    'PNC': 'purpose not cause',
}

ARGS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA']
# ARGMS = None
ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD', 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']

TAGS = ['O', 'B-N', 'B-C', 'B-R', 'I-N', 'I-C', 'I-R']
TAGS2ID = dict([(j, i) for i, j in enumerate(TAGS)])

LABELS = None
LABELS1 = None
ALL_LABELS = None

frames = None


def label2query(label1, label2, _id=None, query_type=None):
    """
    Args:
        label1: ARG or ARGM
        label2: 0-5,A,LOC,RMP,etc.
        _id: frameset id
    """
    if label1 == 'ARG':
        assert len(label2) == 2, (label1, label2, _id)
        desc = ARGS_DESC[label2]
        label2 = label2[-1]
        if query_type == 0:
            query = f'what are the arg{desc} of predicate X?'
        elif query_type == 1:
            # only use meaning
            if (_id in frames) and (label2 in frames[_id]['args']):
                meaning = frames[_id]['args'][label2]
                query = f"what are the arguments of predicate X with meaning {meaning} ?"
            else:
                query = f"what are the arg{desc} of predicate X?"          
        elif query_type == 2:
            if (_id in frames) and (label2 in frames[_id]['args']):
                meaning = frames[_id]['args'][label2]
                query = f"what are the arg{desc} of predicate X with meaning {meaning} ?"
            else:
                query = f"what are the arg{desc} of predicate X?"
        else:
            raise Exception("Invalid query type number")
    else:
        if query_type == 0:
            query = f"what are the {label2} modifiers of predicate X?"
        elif query_type == 1:
            desc = ARGMS_DESC[label2]
            query = f"what are the {desc} modifiers of predicate X?"
        else:
            raise Exception("Invalid query type number")
#     print(query)
    return query


class MyDataset2:
    def __init__(self, path="", tokenizer=None, max_tokens=1024, gold_level=0, arg_query_type=0, argm_query_type=0):
        '''
        gold_level=0: gold predicate disambiguation
        gold_level=1: predict predicate disambiguation

        arg_query_type=0: query with label only
        arg_query_type=1: query without label but with semantics
        arg_query_type=2: query with label and semantics
 
        argm_query_type=0: query with label only
        argm_query_type=1: query with semantics
        '''
        #get initial features
        if not path:
            return
        data = json.load(open(path))
        self.data = copy.deepcopy(data)
        self.input_ids = []
        self.token_type_ids = []
        self.target = []
        self.ids = []
        self.gold = []
        #the evaluation for CONL 2009 includes the results of predicate disambiguation
        self.gold_senses = []
        self.senses = [] #predict sense under predict lemma
        self.batch_input_ids = []
        self.batch_token_type_ids = []
        self.batch_target = []
        self.batch_attention_mask = []
        self.arg_list = []
        self.query_list = []
        self.context_list = []
        self.ids_list = []
        self.gold_list = []
        self.init_data(self.data, tokenizer, max_tokens,
                       gold_level, arg_query_type, argm_query_type)
        

    def init_data(self, data, tokenizer, max_tokens, gold_level, arg_query_type, argm_query_type):
        all_evaluate_obj = []
        for s_id, d in enumerate(tqdm(data, desc='preprocessing')):
            sentence = d['sentence']
            if 'roberta' in tokenizer.name_or_path:
                for i in range(1, len(sentence)):
                    sentence[i] = ' '+sentence[i]

            #print("______________sentence______________")
            #print(sentence)
            sentence1 = [tokenizer.tokenize(w) for w in sentence]
            predicates = d['predicates']
            arguments = d['arguments']
            lemmas = d['lemmas']
            frameset_ids = d['frameset_ids']
            for i in range(len(predicates)):
                evaluate_obj = {}
                pre = predicates[i]
                pre_str = sentence[pre]
                sentence1i = sentence1[:pre]+[['<p>']] + sentence1[pre:pre+1]+[['</p>']]+sentence1[pre+1:]
                args = arguments[i]
                labels = d['plabel'][i]
                labels = [l if len(l)!=4 else 'A'+l[-1] for l in labels ]
                self.gold_senses.append((s_id,pre,lemmas[i],frameset_ids[i]))
                self.senses.append((s_id,pre,d['plemma_ids'][i].split('.')[0],d['plemma_ids'][i].split('.')[1]))
                if gold_level == 0:
                    lem_id = lemmas[i]+'.'+frameset_ids[i]
                elif gold_level == 1:
                    lem_id = d['plemma_ids'][i]
                else:
                    raise Exception(f"Invalid gold level")
                lengthi = [len(s) for s in sentence1i]
                sentence2i = sum(sentence1i, [])
                args1_0 = {('ARG', ar): [] for ar in ARGS}
                args1_1 = {('ARGM', arm): [] for arm in ARGMS}
                args1 = {**args1_0, **args1_1}
                all_gold = []
                for arg in args:
                    s, e, l = arg
                    if 'V' in l.split('-') or ('_' in l.split('-')):
                        continue
                    s = s if s < pre else s+2
                    e = e if e < pre else e+2
                    s1 = sum(lengthi[:s])
                    e1 = sum(lengthi[:e])
                    e2 = e1+lengthi[e]-1
                    l1 = l.split('-')[-1]
                    if len(l1) == 4:
                        l1 = 'A'+l1[-1]
                    l1a = 'ARG' if l1 in ARGS else 'ARGM'
                    if l[0] == 'R':
                        l1b = 'R'
                    elif l[0] == 'C':
                        l1b = 'C'
                    else:
                        l1b = 'N'
                    gold = (s_id, pre, s1, e2, f'{l1b}-{l1a}-{l1}')
                    self.gold.append(gold)
                    all_gold.append(gold)
                    if l1 not in ARGS+ARGMS:
                        continue
                    key = (l1a, l1)
                    args1[key].append((l1b, s1, e2))
                fargs = [k for k in labels if k in ARGS]
                argms = [k for k in labels if k in ARGMS]
                fargs_key = [('ARG', k) for k in fargs]
                argms_key = [('ARGM', k) for k in argms]
                for k in fargs_key+argms_key:
                    v = args1[k]
                    query_type = arg_query_type if k[0] == 'ARG' else argm_query_type
                    q = label2query(k[0], k[1], lem_id,
                                    query_type).replace('X', pre_str)
                    q_tokenize = tokenizer.tokenize(q)
                    if 'roberta' in tokenizer.name_or_path:
                        txt = ['<s>']+q_tokenize+['</s>']+['</s>']+sentence2i+['</s>']
                    else:
                        txt = ['[CLS]']+q_tokenize+['[SEP]']+sentence2i+['[SEP]']
                    txt_ids = tokenizer.convert_tokens_to_ids(txt)
                    txt_ids = torch.tensor(txt_ids, dtype=torch.long)
                    self.input_ids.append(txt_ids)
                    token_type_ids = torch.zeros(
                        txt_ids.shape, dtype=torch.uint8)
                    if 'roberta' in tokenizer.name_or_path:
                        token_type_ids[len(q_tokenize)+3:] = 1
                    else:
                        token_type_ids[len(q_tokenize)+2:] = 1
                    self.token_type_ids.append(token_type_ids)
                    label = ['O' for _ in txt_ids]
                    for ar in v:
                        l, s, e = ar
                        if 'roberta' in tokenizer.name_or_path:
                            s = s+len(q_tokenize)+3
                            e = e+len(q_tokenize)+3
                        else:
                            s = s+len(q_tokenize)+2
                            e = e+len(q_tokenize)+2
                        label[s] = 'B-'+l
                        for j in range(s+1, e+1):
                            label[j] = 'I-'+l
                    target = [TAGS2ID[l] for l in label]
                    target = torch.tensor(target, dtype=torch.uint8)
                    self.target.append(target)
                    self.ids.append((s_id, pre, f'{k[0]}-{k[1]}'))
                    
                    
                    evaluate_obj[k[1]] = {}
                    evaluate_obj[k[1]]["input_ids"] = txt_ids
                    evaluate_obj[k[1]]["token_type_ids"] = token_type_ids
                    evaluate_obj[k[1]]["target"] = target
                    evaluate_obj[k[1]]["query"] = q
                    evaluate_obj[k[1]]["context"] = sentence
                    evaluate_obj[k[1]]["ids"] = (s_id, pre, f'{k[0]}-{k[1]}')
                    evaluate_obj[k[1]]["gold"] = all_gold
                    
                           
                if len(evaluate_obj) != 0:
                    self.do_batch(evaluate_obj)
                    all_evaluate_obj.append(evaluate_obj)

               
             


        
    def do_batch(self, evaluate_obj):
        all_input_ids = [evaluate_obj[key]["input_ids"] for key in evaluate_obj]
        all_token_type_ids = [evaluate_obj[key]["token_type_ids"] for key in evaluate_obj]
        all_targets = [evaluate_obj[key]["target"] for key in evaluate_obj]
        all_query = [evaluate_obj[key]["query"] for key in evaluate_obj]
        all_args = [key for key in evaluate_obj]
        all_contexts = [evaluate_obj[key]["context"] for key in evaluate_obj]
        all_ids = [evaluate_obj[key]["ids"] for key in evaluate_obj]
        all_gold = [evaluate_obj[key]["gold"] for key in evaluate_obj]
        
        all_input_ids_pad = pad_sequence(all_input_ids, batch_first=True)
        all_token_type_ids_pad = pad_sequence(all_token_type_ids, batch_first=True, padding_value=0)
        all_targets_pad = pad_sequence(all_targets, batch_first=True, padding_value=TAGS2ID['O'])
        attention_mask_pad = torch.zeros(all_input_ids_pad.shape, dtype=torch.uint8)
        for i in range(len(all_input_ids)):
                attention_mask_pad[i, :len(all_input_ids[i])] = 1
        
        self.batch_input_ids.append(all_input_ids_pad)
        self.batch_token_type_ids.append(all_token_type_ids_pad)
        self.batch_target.append(all_targets_pad)
        self.batch_attention_mask.append(attention_mask_pad)
        self.arg_list.append(all_args)
        self.query_list.append(all_query)
        self.context_list.append(all_contexts)
        self.ids_list.append(all_ids)
        self.gold_list.append(all_gold)
        
        


    def save(self, save_dir):
        #save processed data
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        input_ids = [d.numpy() for d in self.input_ids]
        token_type_ids = [d.numpy() for d in self.token_type_ids]
        target = [d.numpy() for d in self.target]
        ids = np.array(self.ids)
        gold = np.array(self.gold)
        np.save(os.path.join(save_dir,'gold_senses.npy'),self.gold_senses)
        np.save(os.path.join(save_dir,'senses.npy'),self.senses)
        np.save(os.path.join(save_dir, 'gold.npy'), gold)
        np.save(os.path.join(save_dir, 'input_ids.npy'), input_ids)
        np.save(os.path.join(save_dir, 'token_type_ids.npy'), token_type_ids)
        np.save(os.path.join(save_dir, 'target.npy'), target)
        np.save(os.path.join(save_dir, 'ids.npy'), ids)

    def load(self, save_dir, max_tokens):
        #load cached data
        input_ids = np.load(os.path.join(save_dir, 'input_ids.npy'), allow_pickle=True)
        token_type_ids = np.load(os.path.join(save_dir, 'token_type_ids.npy'), allow_pickle=True)
        target = np.load(os.path.join(save_dir, 'target.npy'), allow_pickle=True)
        ids = np.load(os.path.join(save_dir, 'ids.npy'), allow_pickle=True)
        gold_senses = np.load(os.path.join(save_dir,'gold_senses.npy'),allow_pickle=True)
        senses = np.load(os.path.join(save_dir,'senses.npy'),allow_pickle=True)
        self.gold_senses = [(int(i[0]),int(i[1]),i[2],i[3]) for i in gold_senses]
        self.senses = [(int(i[0]),int(i[1]),i[2],i[3]) for i in senses]
        self.gold = np.load(os.path.join(save_dir, 'gold.npy'), allow_pickle=True).tolist()
        self.gold = [(int(g[0]), int(g[1]), int(g[2]), int(g[3]), g[4]) for g in self.gold]
        self.input_ids = [torch.from_numpy(d) for d in input_ids]
        self.token_type_ids = [torch.from_numpy(d) for d in token_type_ids]
        self.target = [torch.from_numpy(d) for d in target]
        self.ids = ids.tolist()
        self.ids = [(int(i[0]), int(i[1]), i[2]) for i in self.ids]
        self.do_batch(max_tokens)

    def __len__(self):
        return len(self.batch_target)

    def __getitem__(self, i):
        return {'input_ids': self.batch_input_ids[i].long(), 'token_type_ids': self.batch_token_type_ids[i].long(),
                'target': self.batch_target[i].long(), 'attention_mask': self.batch_attention_mask[i].float(),
               'args': self.arg_list[i], "queries": self.query_list[i], "context": self.context_list[i], 
               'ids': self.ids_list[i], "gold": self.gold_list[i]}


def load_data2(path, pretrained_model_name_or_path, max_tokens, shuffle, dataset_tag, local_rank=-1, gold_level=0, arg_query_type=0, argm_query_type=0):
    global frames, LABELS1, ALL_LABELS, ARGMS
    if dataset_tag == 'conll2005' or dataset_tag == 'conll2009':
        ARGMS = ['DIR', 'LOC', 'MNR', 'TMP', 'EXT', 'REC',
                 'PRD', 'PNC', 'CAU', 'DIS', 'ADV', 'MOD', 'NEG']
    elif dataset_tag == 'conll2012':
        ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
                 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']
    else:
        raise Exception("Invalid Dataset Tag:%s" % dataset_tag)
    frames_path = os.path.join(os.path.split(path)[0], 'frames.json')
    frames = json.load(open(frames_path))
    ALL_LABELS = ['O']+[t1+'-'+t0 for t0 in ARGS+ARGMS for t1 in TAGS[1:]]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<p>', '</p>']})
    dataset = MyDataset2(path, tokenizer, max_tokens,
                        gold_level, arg_query_type, argm_query_type)
    sampler = DistributedSampler(
        dataset, rank=local_rank) if local_rank != -1 else None
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            shuffle=shuffle if sampler is None else False, collate_fn=lambda x: x[0])
    return dataloader, dataset


def reload_data(path, max_tokens, shuffle=False, local_rank=-1):
    dataset = MyDataset()
    dataset.load(path, max_tokens)
    sampler = DistributedSampler(
        dataset, rank=local_rank) if local_rank != -1 else None
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            shuffle=shuffle if sampler is None else False, collate_fn=lambda x: x[0])
    return dataloader
