import torch
from transformers import Seq2SeqTrainingArguments

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer

from Utils.code_utils import formulate_code
from Utils.function_calls import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AutoModelForSeq2SeqLM.from_pretrained(
    r"../learn2generate_code/models/marian_with_bert_simAPI_finetune/checkpoint-18870", max_length=512)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    r"../learn2generate_code/models/marian_with_bert_simAPI_finetune/checkpoint-18870", padding='max_length',
    use_fast=True,
    model_max_length=512)

from Utils import evaluator

evaluator = evaluator.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                       max_length=512, padding=True,  ####new
                                       model=model)

from datasets import Dataset, load_metric
import pandas as pd

train_dataset = Dataset.from_csv("../learn_embeddings/data/train.csv")
val_dataset = Dataset.from_csv("../learn_embeddings/data/val.csv")
test_dataset = Dataset.from_csv("../learn_embeddings/data/test.csv")

encoder_length = 512
decoder_length = 512
batch_size = 1


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=encoder_length)
    if batch["answer"][0] == None:
        outputs = tokenizer(['NAA'], padding="max_length", truncation=True, max_length=decoder_length)
    else:
        outputs = tokenizer(batch["answer"], padding="max_length", truncation=True, max_length=decoder_length)
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
train_data = train_dataset.map(  # .select(range(int(len(train_dataset)*0.7)))
    map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['prompt', 'code', 'answer'],
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_data = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=1, remove_columns=['prompt', 'code', 'answer'],
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


class Custom_trainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_bleu = load_metric('bleu')

    def calcBLEU(self, decoded_preds, decoded_labels):

        # Calculate the BLEU scores then return them.
        def bleuTok(arr):
            return list(map(lambda x: x.split(' '), arr))

        bleu_toked_preds = bleuTok(decoded_preds)
        blue_toked_labels = [[x] for x in bleuTok(decoded_labels)]
        return self.normal_bleu.compute(
            predictions=bleu_toked_preds,
            references=blue_toked_labels,
        )

    # map data correctly
    def generate_code(self, model, inputs):
        # inputs.data = {"input_ids":}
        outputs = model.generate(**inputs)

        # all special tokens including will be removed
        output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_str

    def code_output(self, int_pred_code, labels):

        # compute custom loss (suppose one has 3 labels with different weights)
        loss = 0.
        for enc_label, enc_pred_code in zip(labels, int_pred_code):

            # code = self.tokenizer.batch_decode(res, skip_special_tokens=True)

            ref_ans = self.tokenizer.decode(enc_label, skip_special_tokens=True)
            pred_ans = {}
            ref_ans = {"ret_vale": ref_ans}
            try:
                pred_code = self.tokenizer.decode(enc_pred_code, skip_special_tokens=True)
                pred_code = formulate_code(pred_code)
                # int_pred_code = formulate_code(pred_code)
                exec(pred_code, globals(), pred_ans)
            except Exception as e:
                print("could not compute reference answer", e)
                loss += 1.
                continue
                pass
            # print(str(pred_ans["ret_vale"]), str(ref_ans["ret_vale"]))
            max_length = max(len(str(pred_ans["ret_vale"])), len(str(ref_ans["ret_vale"])))
            loss += 1. - 1 / (self.calcBLEU(
                str(pred_ans["ret_vale"]) + "".join(["_"] * (max_length - len(str(pred_ans["ret_vale"]))))
                , str(ref_ans["ret_vale"]) + "".join(["_"] * (max_length - len(str(ref_ans["ret_vale"])))))[
                "precisions"][0])
        return loss / len(labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        original_loss = super(Custom_trainer, self).compute_loss(model, inputs, return_outputs)
        outputs = model.generate(input_ids=inputs.input_ids)
        new_loss = original_loss.detach()
        new_loss.data.fill_(1.0)
        # Sample a token from the predicted probability distribution
        code_output_ = self.code_output(outputs, inputs.data["labels"])
        loss = original_loss * code_output_

        return loss


training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints/RL_trian_marian_finetuned_with_bert_embeds",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    num_train_epochs=5,
    do_train=True,
    do_eval=True,
    fp16=True,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    seed=1995,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Custom_trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=evaluator,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
