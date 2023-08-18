import torch

torch.cuda.empty_cache()

import re
from datetime import datetime
from time import sleep
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, MarianConfig, TrainingArguments, Trainer, AutoTokenizer, \
    AutoModelForSeq2SeqLM, AutoModelForMaskedLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
from Utils import evaluator

device = torch.device("cuda")
print("using ", device)

tokenizer = AutoTokenizer.from_pretrained("arrafmousa/SimAPI_tokenizer", use_fast=True)

model = AutoModelForSeq2SeqLM.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large", ignore_mismatched_sizes=True)

embedding_layer = AutoModelForMaskedLM.from_pretrained(
    r"../learn_embeddings/models/bert_embedder").bert.embeddings.word_embeddings
model.model.shared = embedding_layer
model.model.encoder.embed_tokens = embedding_layer
model.model.decoder.embed_tokens = embedding_layer
# model.config.bad_words_ids = [[3]]
# model.config.decoder_start_token_id = 0
# model.config.pad_token_id = 1
model.resize_token_embeddings(len(tokenizer))


def create_dataset_from_raw(path):
    df = pd.read_csv(path)

    dataset = {'answers': [], 'prompt': [], 'id': []}
    for idx, entry in df.iterrows():
        dataset['answers'].append(entry["code"])
        dataset["prompt"].append(entry["context"] + " " + entry["question"])
        dataset["id"].append(idx)

    raw_dataset = Dataset.from_dict(dataset)
    return raw_dataset


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
    elif dataset_type == "astropy" or dataset_type == "simAPI":
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


print("loading datasets")

train_astropy = create_dataset("../learn_embeddings/data/train.csv", dataset_type="simAPI")
val_astropy = create_dataset("../learn_embeddings/data/val.csv", dataset_type="simAPI")

batch_size = 4

train_dl = DataLoader(train_astropy, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_astropy, batch_size=batch_size, shuffle=True)

encoder_length = 512
decoder_length = 512
batch_size = 1


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    inputs = tokenizer(batch["prompt"],
                       padding="max_length", truncation=True,
                       max_length=encoder_length)
    outputs = tokenizer(batch["answers"], padding="max_length", truncation=True, max_length=decoder_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask

    """
    # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
    batch["labels"] = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
    ]
    """
    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch


# make train dataset ready
train_data = train_astropy.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['answers', 'prompt'],
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_data = val_astropy.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['answers', 'prompt'],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

evaluator = evaluator.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                       max_length=512, padding=True,  ####new
                                       model=model)

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="models/marian_with_bert_simAPI_finetune",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    num_train_epochs=30,
    do_train=True,
    do_eval=True,
    fp16=True,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    weight_decay=0.0000005,
    seed=1995,
    load_best_model_at_end=True,
    metric_for_best_model='eval_BLEU',
)
#

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=evaluator,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
