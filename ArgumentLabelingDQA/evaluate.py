import json
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import pipeline

from dataloader import *

zero_shot = True


def get_score(gold_set, predict_set):
    TP = len(set.intersection(gold_set, predict_set))
    precision = TP / (len(predict_set) + 1e-9)
    recall = TP / (len(gold_set) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def get_ans(arg_label, frame_key, core_info, arg_dict, context, qa_pipeline, use_naive_query):
    if use_naive_query:
        question = label2query_naive(arg_label, frame_key, core_info, arg_dict)
    else:
        question = label2query(arg_label, frame_key, core_info, arg_dict)
    sample = {
        'context': context,
        'question': question,
    }
    res = qa_pipeline(sample, handle_impossible_answer=True)
    if res['end'] != 0 and res['answer'] not in ('a', 'the'):
        sample['answer'] = res
        # print(arg_label, frame_key)
        # print(json.dumps(sample, indent=4))
        if arg_label in ('ARG0', 'ARG1'):
            arg_dict[arg_label] = ([res['start']], [res['end']], [res['answer']])
        return res


def eval_for_sen(sen_id, sen_dict, gold_level, qa_pipeline, use_naive_query):
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
    golds = {}
    predicts = {}
    for i in range(len(predicates)):
        pred_id = predicates[i]
        p_label = p_labels_list[i]
        arguments = arguments_list[i]
        frameset_id = frameset_ids[i]
        lemma = lemmas[i]
        frame_key = '.'.join([lemma, frameset_id])
        for (s_id, e_id, arg_label) in arguments:
            if arg_label != 'V' and not (zero_shot and arg_label[0] in ('R', 'C')):
                if (sen_id, i, arg_label) not in golds:
                    golds[(sen_id, i, arg_label)] = []
                golds[(sen_id, i, arg_label)].append(
                    (id2pos[s_id][0], id2pos[e_id][1], context[id2pos[s_id][0]: id2pos[e_id][1]]))

        # get argument candidates
        core_arg_list = []
        other_arg_list = []
        if zero_shot:
            prefixes = ['']
        else:
            prefixes = ['', 'R-', 'C-']
        for p_arg in p_label:
            for prefix in prefixes:
                if len(p_arg) == 3:
                    prefix += 'ARGM-'
                p_arg_l = prefix + p_arg
                if p_arg_l in ('ARG0', 'ARG1'):
                    core_arg_list.append(p_arg_l)
                else:
                    other_arg_list.append(p_arg_l)

        arg_dict = {}
        for arg_label in core_arg_list:
            res = get_ans(arg_label, frame_key, '', arg_dict, context, qa_pipeline, use_naive_query)
            if res is not None:
                predicts[(sen_id, i, arg_label)] = res
            # if (sen_id, i, arg_label) in golds:
            #     print(arg_label, 'gold answer:', ';'.join([text for _, _, text in golds[(sen_id, i, arg_label)]]))

        # cal core info
        arg_dict['V'] = ([id2pos[pred_id][0]], [id2pos[pred_id][1]], [sentence[pred_id]])
        arg_sort_list = sorted(arg_dict.items(), key=lambda x: x[1][0][0])
        core_args = [(s_pos, arg_text) for arg_label, (s_poses, _, arg_texts) in arg_sort_list if
                     arg_label in ('V', 'ARG0', 'ARG1') for s_pos, arg_text in zip(s_poses, arg_texts)]
        sorted_core_args = sorted(core_args, key=lambda x: x[0])
        core_info = ' '.join(list(zip(*sorted_core_args))[1])
        del arg_dict['V']

        for arg_label in other_arg_list:
            if arg_label[1] == '-' and arg_label.split('-')[1] not in arg_dict:
                continue
            res = get_ans(arg_label, frame_key, core_info, arg_dict, context, qa_pipeline, use_naive_query)
            if res is not None:
                predicts[(sen_id, i, arg_label)] = res
            # if (sen_id, i, arg_label) in golds:
            #     print(arg_label, 'gold answer:', ';'.join([text for _, _, text in golds[(sen_id, i, arg_label)]]))
    return golds, predicts


def evaluation(model, data_path, args):
    global zero_shot
    if 'bert' not in args.checkpoint_path:
        zero_shot = False
    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(torch.device(device_s))
    device_id = 0 if device_s.startswith('cuda') else -1
    qa_pipeline = pipeline('question-answering', model=model, tokenizer='deepset/roberta-large-squad2',
                           device=device_id)
    data = json.load(open(data_path))
    golds_dict = {}
    predicts_dict = {}

    for s_id, d in enumerate(tqdm(data, desc='evaluating')):
        s_golds, s_predicts = eval_for_sen(s_id, d, args.gold_level, qa_pipeline, args.naive_query)
        golds_dict.update(s_golds)
        predicts_dict.update(s_predicts)

    golds_root = set()
    predicts_root = set()
    for key, pred_res in tqdm(predicts_dict.items(), desc='calculating'):
        if key in golds_dict:
            for g_s, g_e, text in golds_dict[key]:
                gold_root = {get_root(text), get_noun_root(text)}
                pred_root = {get_root(pred_res['answer']), get_noun_root(pred_res['answer'])}
                if len(set.intersection(gold_root, pred_root)) > 0:
                    predicts_root.add((key, g_s, g_e))
                    golds_root.add((key, g_s, g_e))
                    break
            else:
                predicts_root.add((key, pred_res['start'], pred_res['end']))
                golds_root.add((key, golds_dict[key][0][0], golds_dict[key][0][1]))
        else:
            predicts_root.add((key, pred_res['start'], pred_res['end']))
    predicts = {(key, r['start'], r['end']) for key, r in predicts_dict.items()}
    golds = {(key, r[0][0], r[0][1]) for key, r in golds_dict.items()}
    print('root match:')
    get_res(golds_root, predicts_root)
    print('strict match:')
    return get_res(golds, predicts)


def get_res(golds, predicts):
    gold_dict = defaultdict(set)
    for gold_item in golds:
        arg_l = gold_item[0][-1]
        if 'ARG0' in arg_l or 'ARG1' in arg_l:
            if arg_l[0] in ('R', 'C'):
                gold_dict[arg_l[:5]].add(gold_item)
            else:
                gold_dict['N-ARG'].add(gold_item)
        else:
            if arg_l[0] in ('R', 'C'):
                gold_dict[arg_l[:2] + 'other'].add(gold_item)
            else:
                gold_dict['N-other'].add(gold_item)
    gold_N_arg01 = gold_dict['N-ARG']
    gold_R_arg01 = gold_dict['R-ARG']
    gold_C_arg01 = gold_dict['C-ARG']
    gold_N_arg_other = gold_dict['N-other']
    gold_R_arg_other = gold_dict['R-other']
    gold_C_arg_other = gold_dict['C-other']

    pred_dict = defaultdict(set)
    for pred_item in predicts:
        arg_l = pred_item[0][-1]
        if 'ARG0' in arg_l or 'ARG1' in arg_l:
            if arg_l[0] in ('R', 'C'):
                pred_dict[arg_l[:5]].add(pred_item)
            else:
                pred_dict['N-ARG'].add(pred_item)
        else:
            if arg_l[0] in ('R', 'C'):
                pred_dict[arg_l[:2] + 'other'].add(pred_item)
            else:
                pred_dict['N-other'].add(pred_item)
    pred_N_arg01 = pred_dict['N-ARG']
    pred_R_arg01 = pred_dict['R-ARG']
    pred_C_arg01 = pred_dict['C-ARG']
    pred_N_arg_other = pred_dict['N-other']
    pred_R_arg_other = pred_dict['R-other']
    pred_C_arg_other = pred_dict['C-other']
    combines = [['N-arg 0/1', gold_N_arg01, pred_N_arg01],
                ['N-arg oth', gold_N_arg_other, pred_N_arg_other],
                ['R-arg 0/1', gold_R_arg01, pred_R_arg01],
                ['R-arg oth', gold_R_arg_other, pred_R_arg_other],
                ['C-arg 0/1', gold_C_arg01, pred_C_arg01],
                ['C-arg oth', gold_C_arg_other, pred_C_arg_other]]
    for name, gold_part, pred_part in combines:
        p, r, f = get_score(gold_part, pred_part)
        print(name + ':', 'p:%.4f' % p, 'r:%.4f' % r, 'f:%.4f' % f)
    p, r, f = get_score(golds, predicts)
    print('global:', 'p:%.4f' % p, 'r:%.4f' % r, 'f:%.4f' % f)
    return {"p": p, "r": r, "f": f}