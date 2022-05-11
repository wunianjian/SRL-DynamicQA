import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from dataloader2 import *
from train import *

ARGS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA']
ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
                 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']


def get_score(gold_set, predict_set):
    TP = len(set.intersection(gold_set, predict_set))
#     print("TP: ", TP)
#     print("GOLD SET: ", gold_set)
#     print("PREDICT SET: ", predict_set)
    #print("#TP",TP,"#gold",len(gold_set),"#predict",len(predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision, recall, f1


def get_core_if_possible(batch):
    input_ids, token_type_ids, attention_mask, query, arg_list, context, ids, gold = batch['input_ids'], \
                batch['token_type_ids'], batch['attention_mask'], batch['queries'], batch['args'], batch["context"], \
                batch["ids"], batch["gold"]
    
    has_core = False
    if "A0" in arg_list or "A1" in arg_list:
        has_core = True
        if "A0" not in arg_list:
            core_input_ids = input_ids[arg_list.index("A1")].reshape(1, -1)
            core_token_type_ids = token_type_ids[arg_list.index("A1")].reshape(1, -1)
            core_attention_mask = attention_mask[arg_list.index("A1")].reshape(1, -1)
            noncore_query = [query[i] for i in range(len(query)) if i != arg_list.index("A1")]
        elif "A1" not in arg_list:
            core_input_ids = input_ids[arg_list.index("A0")].reshape(1, -1)
            core_token_type_ids = token_type_ids[arg_list.index("A0")].reshape(1, -1)
            core_attention_mask = attention_mask[arg_list.index("A0")].reshape(1, -1)
            noncore_query = [query[i] for i in range(len(query)) if i != arg_list.index("A0")]
        else:
            core_input_ids = torch.cat((input_ids[arg_list.index("A0")], input_ids[arg_list.index("A1")])).reshape(-1, input_ids.shape[1])
            core_token_type_ids = torch.cat((token_type_ids[arg_list.index("A0")], token_type_ids[arg_list.index("A1")])).reshape(-1, token_type_ids.shape[1])
            core_attention_mask = torch.cat((attention_mask[arg_list.index("A0")], attention_mask[arg_list.index("A1")])).reshape(-1, attention_mask.shape[1])
            noncore_query = [query[i] for i in range(len(query)) if i != arg_list.index("A1") and i != arg_list.index("A0")]
        return has_core, core_input_ids, core_token_type_ids, core_attention_mask, noncore_query, context, ids, gold
    
    else:
        return has_core, input_ids, token_type_ids, attention_mask, query, context, ids, gold


# +
def compute_sep_predicts(sep_predicts, ids, gold):
    sep_predicts1 = []
    for p, _id in zip(sep_predicts, ids):
        s_id, p_id, l0 = _id
        for pi in p:
            s, e, l1 = pi
            sep_predicts1.append((int(s_id), int(p_id), s, e, l1+'-'+l0))          
    return sep_predicts1
          

def compute_glb_predicts(predict_probs, context_masks, ids, gold):
    # get all (sentence_id,predicate_id)
    ids1 = [i[:2] for i in ids]
    ids1 = list(set(ids1))
    ids12idx = {k: v for v, k in enumerate(ids1)}
    # get all roles
    ids2 = ['ARG-'+i for i in ARGS]+['ARGM-'+i for i in ARGMS]
    ids22idx = {k: v for v, k in enumerate(ids2)}
    #predict probality matrix with shape (len(ids1),len(ids2))
    predict_probs1 = [[None for j in range(len(ids22idx))] for i in ids1]
    for p, cm, _id in zip(predict_probs, context_masks, ids):
        p = p[cm]
        i = ids12idx[_id[:2]] 
        j = ids22idx[_id[-1]]
        predict_probs1[i][j] = p
    import copy
    predict_probs2 = copy.deepcopy(predict_probs1)
    predict_probs3 = [None for _ in predict_probs2]
    pad_prob = torch.zeros(7)
    pad_prob[TAGS2ID['O']] = 1
    assert TAGS2ID['O'] == 0 
    for i, p in enumerate(predict_probs1):
        lens = []
        pad_js = []  # record padding index
        for j, pj in enumerate(p):
            if pj is not None:
                lens.append(len(pj))
            else:
                pad_js.append(j)
        if len(lens) > 0:
            assert all([le == lens[0] for le in lens])
        for j in pad_js:
            predict_probs2[i][j] = pad_prob.unsqueeze(0).expand(lens[0], 7) #(seq_len,7)
        # probabilities of tag O
        Os = [p2ij[:, 0] for p2ij in predict_probs2[i]]
        O_prob = torch.ones(Os[0].shape)  # (seq_len,)
        for op in Os:
            O_prob = O_prob*op
        # merge scores
        Ps = [p2ij[:, 1:] for p2ij in predict_probs2[i]]
        Ps = torch.cat(Ps, 1)
        Ps = torch.cat([O_prob.unsqueeze(1), Ps], 1)
        predict_probs3[i] = Ps

    predicts = []
    for p, _id in zip(predict_probs3, ids1):
        s_id, p_id = _id
        p1 = decode(p, dataset_tag)
        for p1i in p1:
            s, e, l = p1i
            assert s >= 0 and e >= 0
            item = (s_id, p_id, s, e, l)
            predicts.append(item)
    return predicts


# +
def evaluation(model, dataloader, amp=False, device=torch.device('cuda'),dataset_tag=''):
    if dataset_tag == 'conll2005' or dataset_tag == 'conll2009':
        ARGMS = ['DIR', 'LOC', 'MNR', 'TMP', 'EXT', 'REC',
                 'PRD', 'PNC', 'CAU', 'DIS', 'ADV', 'MOD', 'NEG']
    elif dataset_tag == 'conll2012':
        ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
                 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']
    else:
        raise Exception("Invalid Dataset Tag:%s" % dataset_tag)  
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    model.to(device)
    tqdm_dataloader = tqdm(dataloader, desc='eval')
#     gold = dataloader.dataset.gold
#     print(gold)
#     ids = dataloader.dataset.ids
#     print(ids)
#     print(len(ids))
    predict_probs = []
    context_masks = []
    sep_predicts = []
    id_list = []
    gold_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm_dataloader):
            has_core, input_ids, token_type_ids, attention_mask, \
                noncore_query, context, ids, gold = get_core_if_possible(batch)
            input_ids, token_type_ids, attention_mask = input_ids.to(
                device), token_type_ids.to(device), attention_mask.to(device) 
            if amp:
                with autocast():
                    predict_prob, context_mask = model(
                        input_ids, token_type_ids, attention_mask)
            else:
                predict_prob, context_mask = model(
                    input_ids, token_type_ids, attention_mask)
            
            if has_core:
                predicts = compute_glb_predicts(predict_prob, context_mask, ids, gold)
            
            id_list.extend(list(ids))
            gold_list.extend(list(gold))
            predict_probs.extend(list(predict_prob))
            context_masks.extend(list(context_mask))
            sep_predicts.extend(sep_decode(predict_prob, context_mask))
            
    
    sep_predicts1 = compute_sep_predicts(sep_predicts, id_list, gold_list)
    sp, sr, sf = get_score(set(gold), set(sep_predicts1))
    print("sep score: ", 'p:%.4f'%sp, 'r:%.4f'%sr, 'f:%.4f'%sf)
    
    predicts = compute_glb_predicts(predict_probs, context_masks, id_list, gold_list)
    p, r, f = get_score(set(gold), set(predicts))
    print("glb score: ", 'p:%.4f'%p, 'r:%.4f'%r, 'f:%.4f'%f)
    return {"p":p,"r":r,"f":f}
    

# -

def get_index(p, k, i=0, d=float('inf')):
    if k in p[i:]:
        return p.index(k, i)
    else:
        return d


def decode(predict, dataset_tag):
    if dataset_tag == 'conll2005' or dataset_tag == 'conll2009':
        ARGMS = ['DIR', 'LOC', 'MNR', 'TMP', 'EXT', 'REC',
                 'PRD', 'PNC', 'CAU', 'DIS', 'ADV', 'MOD', 'NEG']
    elif dataset_tag == 'conll2012':
        ARGMS = ['MNR', 'ADV', 'LOC', 'TMP', 'PRP', 'PRD', 'DIR', 'DIS', 'MOD',
                 'NEG', 'CAU', 'EXT', 'LVB', 'REC', 'ADJ', 'GOL', 'DSP', 'PRR', 'COM', 'PRX', 'PNC']
    else:
        raise Exception("Invalid Dataset Tag:%s" % dataset_tag)
    ALL_LABELS = ['O']+[t1+'-'+t0 for t0 in ['ARG-'+i for i in ARGS]+['ARGM-'+i for i in ARGMS] for t1 in TAGS[1:]]
    predict = predict.unsqueeze(0)
    predict = predict.argmax(dim=-1)  # (1,seq_len)
    res = []
    for p in predict:
        s = []
        p = [ALL_LABELS[i] for i in p]
        p1 = [i.split('-')[0] if '-' in i else i for i in p]
        p2 = [i[2:] if '-' in i else i for i in p]
        if 'B' not in p1:
            res.append(s)
            continue
        i = get_index(p1, 'B') 
        x = p2[i]
        while i < len(p)-1:
            for j in range(i+1, len(p)):
                if p[j] != 'I-'+x:
                    s.append((i, j-1, x))
                    break
            if p[j] == 'O':
                if j == len(p)-1 or ('B' not in p1[j+1:]):
                    break
                else:
                    i = get_index(p1, 'B', j+1)
                    x = p2[i]
            else:
                i = j
                x = p2[i]
        res.append(s)
    return res[0]


def sep_decode(predict, context_mask):
    res = []
    predict = predict.argmax(dim=-1)
    for p, cm in zip(predict, context_mask):
        s = []
        p = p[cm]
        p = [TAGS[i] for i in p]
        p1 = [i.split('-')[0] if '-' in i else i for i in p]
        p2 = [i[2:] if '-' in i else i for i in p]
        if 'B' not in p1:
            res.append(s)
            continue
        i = get_index(p1, 'B') 
        x = p2[i]
        while i < len(p)-1:
            for j in range(i+1, len(p)):
                if p[j] != 'I-'+x:
                    s.append((i, j-1, x))
                    break
            if p[j] == 'O':
                if j == len(p)-1 or ('B' not in p1[j+1:]):
                    break
                else:
                    i = get_index(p1, 'B', j+1)
                    x = p2[i]
            else:
                i = j
                x = p2[i]
        res.append(s)
    return res


# +
dev_path = "../../data/conll2012/dev.english.psense.plabel.conll12.json"
pretrained_model_name_or_path = "roberta-large"
max_tokens = 1024
dataset_tag = "conll2012"
local_rank = -1
gold_level = 1
arg_query_type = 2
argm_query_type = 1

dev_dataloader, dev_dataset = load_data2(dev_path, pretrained_model_name_or_path, max_tokens, False,
                                   dataset_tag, local_rank, gold_level, arg_query_type, argm_query_type)

# +
args = args_parser()
model = MyModel(args)
model_state_dict = torch.load("/home/DependencyMRC-SRL/checkpoints/conll2012/arg_labeling/2022_04_30_21_54_02/checkpoint_19.cpt",\
                             map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict["model_state_dict"])
device = args.local_rank if args.local_rank != -1 else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
model.to(device)

score = evaluation(model, dev_dataloader, args.amp, device, args.dataset_tag)
# -

# !python evaluate2.py \
# --dataset_tag conll2012 \
# --pretrained_model_name_or_path roberta-large \
# --train_path ./data/conll2012/train.english.psense.plabel.conll12.json \
# --dev_path ./data/conll2012/dev.english.psense.plabel.conll12.json  \
# --max_tokens 1024 \
# --max_epochs 20 \
# --lr 1e-5 \
# --max_grad_norm 1 \
# --warmup_ratio 0.01 \
# --arg_query_type 2 \
# --argm_query_type 1 \
# --gold_level 1 \
# --tensorboard \
# --eval \
# --save \
# --amp \
# --tqdm_mininterval 50

dev_dataset[0]

dev_dataset[0]["input_ids"][1].reshape(1, -1).shape


