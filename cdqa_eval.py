import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, AutoModel
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import random
import re
import transformers
from typing import List, Mapping, NewType, Optional, Tuple, Union
import time
import json
from collections import Counter
import jieba
from xzk_tasks import *
from xzk_utils import prepare_requests, cut_and_normalize_strs, stop_sequences_criteria, prepare_requests_v3
from vllm import LLM, SamplingParams
from pathlib import Path
import openai, requests

def process_string(s):
    words = []
    for word in ' '.join(jieba.cut(s)).split():
        if word not in '，、。 ,.《》':
            words.append(word)
    return words

def compute_acc_single(gold_toks, pred_toks):
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    return num_same / len(gold_toks)

def compute_acc(a_golds, a_pred, lang):
    if a_pred == '':
        return 0
    golds_toks = [process_string(a_gold) for a_gold in a_golds]
    pred_toks = process_string(a_pred)

    return max(
        compute_acc_single(gold_toks, pred_toks) for gold_toks in golds_toks)

def openai_model(model, messages, max_tokens=1):
    token = os.getenv('gptkey')
    url = "<your-oai-request-url>"
    Headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    payload = {
            "temperature": 0,
            "top_p": 1,
            "model": model,
            "messages": [
                # {'role': 'system', 'content': system}, 
                ]+[{"role": "user" if idx%2==0 else 'assistant', "content": m} for idx, m in enumerate(messages)],
            "max_tokens": max_tokens,
            "n": 1,
            # "stop": ['\n', '\n\n']
        }
    answer = None
    while answer is None:
        try:
            response = requests.post(url, json=payload, headers=Headers)
            res = response.json()
            if res['code'] != 200:
                print(res)
            else:
                # answer = res['data']['response'][0]['content']
                answer = res['data']['response']['choices'][0]['message']['content']
        except Exception as e:
            print(e)
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_openai', action="store_true")
    parser.add_argument('--model_path', type=str, default='Llama-2-7b-hf')
    parser.add_argument("--write_metrics", action="store_true")
    parser.add_argument('--ntrain', type=int, default=0)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default='./experiments')
    parser.add_argument('--task', type=str, default='cfreshqa_fast')
    parser.add_argument('--prompt_type', type=str, default='vanilla') # vanilla, cot, rar
    parser.add_argument('--max_tokens', type=int, default=50)
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(key, ":", value)


    # modify this and out_dir
    if args.task == 'cfreshqa_fast':
        task = cfreshqa('fast changing', args.prompt_type)
    elif args.task == 'cfreshqa_slow':
        task = cfreshqa('slow changing', args.prompt_type)
    elif args.task == 'cfreshqa_never':
        task = cfreshqa('never changing', args.prompt_type)
    elif args.task == 'cfreshqa_fast_google':
        task = cfreshqa_google('fast changing', args.prompt_type)
    elif args.task == 'cfreshqa_slow_google':
        task = cfreshqa_google('slow changing', args.prompt_type)
    elif args.task == 'cfreshqa_never_google':
        task = cfreshqa_google('never changing', args.prompt_type)
    elif args.task == 'cfreshqa_fast_bing':
        task = cfreshqa_bing('fast changing', args.prompt_type)
    elif args.task == 'cfreshqa_slow_bing':
        task = cfreshqa_bing('slow changing', args.prompt_type)
    elif args.task == 'cfreshqa_never_bing':
        task = cfreshqa_bing('never changing', args.prompt_type)
    elif args.task == 'cfreshqa_fast_quark':
        task = cfreshqa_quark('fast changing', args.prompt_type)
    elif args.task == 'cfreshqa_slow_quark':
        task = cfreshqa_quark('slow changing', args.prompt_type)
    elif args.task == 'cfreshqa_never_quark':
        task = cfreshqa_quark('never changing', args.prompt_type)
    else:
        raise NotImplementedError()

    inference_set = prepare_requests_v3(task, args)
    print(inference_set[0]['requests'])

    requests = inference_set['requests']

    if args.is_openai:
        model = LLM(args.model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
        sampling_strategy = SamplingParams(max_tokens=args.max_tokens, temperature=0, stop=[',', '.', '。', '，', '<|endoftext|>'])
        res = model.generate(requests, sampling_strategy)
    else:
        res = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            tasks = [executor.submit(openai_model, args.model_path, [d], args.max_tokens) for d in requests]
            for idx, pred in enumerate(tqdm(tasks, total=len(tasks))):
                pred_text = pred.result()
                res.append({
                    'raw_pred': pred_text,
                    'request_id': idx,
                })

    acc_list = []
    records = []
    eff = 0
    all_answers = inference_set['answers']
    for i in tqdm(range(len(res)), desc="Processed answer"):
        raw_pred = res[i].outputs[0].text if args.is_openai else res[i]['raw_pred']
        pred = cut_and_normalize_strs(raw_pred)
        answers = [cut_and_normalize_strs(ans) for ans in all_answers[int(res[i]['request_id'])]]
        records.append({'raw_pred':raw_pred, 'normalized_pred':pred, 'answers':answers, 'requests':requests[i]})
        error = ['对不起', '抱歉', '无法确定', '没有提及']
        if any([e in pred for e in error]): 
            continue
        eff += 1
        acc = compute_acc(a_golds=answers, a_pred=pred, lang='zh')
        acc_list.append(acc)


    f1_score = format(sum(acc_list)/len(acc_list)*100, '.4f')
    answer_rate = round(100*eff/len(res), 2)
    print(f"F1(answer rate):{f1_score}({answer_rate}%)")


    if args.write_metrics:
        os.makedirs(os.path.join(args.out_dir,args.task), exist_ok=True)
        if '/' in args.model_path:
            model_name = args.model_path.split('/')[-1]
        else:
            model_name = args.model_path
        cur_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        out_metrics = os.path.join(args.out_dir,args.task , f'{cur_time}--{model_name}--{args.prompt_type}--{args.ntrain}shots--{args.k}k--{answer_rate}AR--{f1_score}F1.metrics')
        with open(out_metrics, 'w') as f:
            for key, value in vars(args).items():
                f.writelines(f"{key}:{value}\n")
            f.write(inference_set[9]['requests'])
        out_predict_file = os.path.join(args.out_dir,args.task , f'{cur_time}--{model_name}--{args.prompt_type}--{args.ntrain}shots--{args.k}k.prediction')
        with open(out_predict_file, 'w') as f:
            json.dump(records, f, indent=6, ensure_ascii=False)


if __name__ == "__main__":
    main()