from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
import pandas as pd
import io

# Load the training and testing datasets using pandas
df1 = pd.read_csv("") #training dataset
df2 = pd.read_csv("") #testing dataset
df1 = Dataset.from_pandas(df1)
df2 = Dataset.from_pandas(df2)

# Convert the pandas DataFrames to dictionaries
train_data = {"train": df1}
test_data = {"test": df2}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

special_tokens_dict = {'additional_special_tokens': ['[FEATURE1]', '[FEATURE2]','[FEATURE3]']}
tokenizer.add_special_tokens(special_tokens_dict)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)  # Change num_labels based on your classification task

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")

tokenized_dataset = df1.map(tokenize_batch, batched=True)
tokenized_dataset2 = df2.map(tokenize_batch, batched=True)

def inspect_tokenization(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_dataset2.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
model.to("cuda")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Specify the metric to use for early stopping
    greater_is_better=False  # Set to True if a higher metric score is better
)

from transformers import DataCollatorWithPadding, EarlyStoppingCallback

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset2,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping callback
)

trainer.train()

model.save_pretrained("") # your location and file name
tokenizer.save_pretrained("") # your location and file name