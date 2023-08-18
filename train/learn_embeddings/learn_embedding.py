import re

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling, Trainer, AutoConfig, BertModel, BertConfig

config = BertConfig.from_pretrained("distilroberta-base")
config.num_attention_heads = 8
new_embedding_size = 512  # Example new embedding size
config.hidden_size = new_embedding_size
config.intermediate_size = 4 * new_embedding_size  # Adjust intermediate size accordingly
model = AutoModelForCausalLM.from_config(config)
model.load_state_dict(model.state_dict())

tokenizer = AutoTokenizer.from_pretrained("AhmedSSoliman/MarianCG-CoNaLa-Large")
count_threshold = 200


def get_new_tokens():
    def count_word_occurrences(text):
        # Convert the text to lowercase and split it into words
        delimiters = [",", ";", ".", "\"", "\'", "*", "[EOL]", "(", ")", "\\", "_", " ", "=", "/"]
        pattern = '|'.join(map(re.escape, delimiters))
        words = re.split(pattern, text)

        # Create an empty dictionary to store word counts
        word_counts = {}

        # Iterate through the words and update the counts
        for word in words:
            # Increment the count if the word is already in the dictionary
            if word in word_counts:
                word_counts[word] += 1
            # Otherwise, initialize the count to 1
            else:
                word_counts[word] = 1

        return word_counts

    # Example usage:
    text = ''.join(pd.read_csv(r"data/train.csv")["code"])
    counts = count_word_occurrences(text)

    # Display the word counts
    new_tokens = []
    for word, count in counts.items():
        if count > count_threshold:
            new_tokens.append(word)

    return new_tokens


new_tokens = get_new_tokens()
new_tokens = set(new_tokens) - set(tokenizer.encoder.keys())
print("adding ", len(new_tokens), " new tokens")
tokenizer.add_tokens(list(new_tokens))
model.resize_token_embeddings(len(tokenizer))


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=512)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=512)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(
    r"data/emb_train.txt",
    r"data/emb_test.txt", tokenizer)

print(train_dataset)
model.train()

training_args = TrainingArguments(
    output_dir="models/bert_embedder",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=20,  # number of training epochs
    per_device_train_batch_size=4,  # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
tokenizer.push_to_hub("SimAPI_tokenizer")
trainer.train()
trainer.save_model(r"models/bert_embedder")
