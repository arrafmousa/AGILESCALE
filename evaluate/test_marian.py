import json
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from Utils.code_utils import formulate_code
from Utils.function_calls import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using ", device)


def create_dataset(path, start_idx=0, dataset_type="conala"):
    if dataset_type == "conala":
        conala_corpus = json.load(open(path))

        dataset = {'answers': [], 'prompt': [], 'id': []}
        for idx, entry in enumerate(conala_corpus):
            if entry["rewritten_intent"] is None:
                dataset["prompt"].append(entry["intent"])
            else:
                dataset["prompt"].append(entry["rewritten_intent"])

            dataset['answers'].append(entry["snippet"])
            dataset["id"].append(idx + start_idx)
        raw_dataset = Dataset.from_dict(dataset)
        return raw_dataset
    elif dataset_type == "astropy":
        astropy_dataset = pd.read_csv(path)
        dataset = {'answers': [], 'prompt': [], 'id': []}
        for idx, row in astropy_dataset.iterrows():
            dataset['answers'].append(row["code"])
            dataset["id"].append(idx + start_idx)
            dataset["prompt"].append(row["prompt"])
        raw_dataset = Dataset.from_dict(dataset)
        return raw_dataset
    else:
        print("dataset type must be conala or astropy")
        exit(-1)


test_simapi = create_dataset("datasets/simapi_test.csv", dataset_type="astropy")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    r"C:\Users\USER\Desktop\AGILESCALE\train\generating_final_answer\checkpoints\RL_trian_marian_finetuned_with_bert_embeds\checkpoint-25150").to(device)
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\Users\USER\Desktop\AGILESCALE\train\generating_final_answer\checkpoints\RL_trian_marian_finetuned_with_bert_embeds\checkpoint-25150",
    use_fast=True)


def generate_new_code(batch):
    inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True,
                       max_length=512, return_tensors="pt")

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # batch["pred_code"] = pred_code
    pred_ans = {}
    ref_ans = {}

    try:
        pred_code = formulate_code(output_str[0])
        print(pred_code)
        exec(pred_code, globals(), pred_ans)
    except Exception as e:
        pred_ans["ret_vale"] = 'NAA-pred'
        print(e, 'NAA-pred')
    # if ref_ans["ret_vale"] == pred_ans["ret_vale"]:
    #     print("ok")

    try:
        exec(formulate_code(batch["answers"][0]), globals(), ref_ans)
        # print(pred_ans)
    except Exception as e:
        ref_ans["ret_vale"] = 'NAA-ref'
        print(e, 'NAA-ref')

    batch["correct_answer"] = [ref_ans["ret_vale"] == pred_ans["ret_vale"]]
    print(batch["correct_answer"])

    return batch


correct = 0
results = test_simapi.map(generate_new_code, batched=True, batch_size=1)
for res in results:
    if res["correct_answer"]:
        correct += 1

print(correct)
print("the accuracy is : ", correct / len(results))
