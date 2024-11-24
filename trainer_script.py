import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, Dataset, DataCollatorForLanguageModeling

class CsvDataset(Dataset):
    def __init__(self, dataframe, tokenizer, input_column, output_column, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.input_column = input_column
        self.output_column = output_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row[self.input_column]
        output_text = row[self.output_column]
        full_text = f"{input_text} \n {output_text}"
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": tokens["input_ids"].squeeze(0),
        }

def main():
    # Define paths
    train_file = "./data/train.csv"  # Path to your training dataset
    eval_file = "./data/eval.csv"    # Path to your evaluation dataset

    # Load the tokenizer and model
    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Resize the token embeddings in case of added special tokens
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    train_df = pd.read_csv(train_file)
    eval_df = pd.read_csv(eval_file)

    train_dataset = CsvDataset(train_df, tokenizer, input_column="hateSpeech", output_column="counterSpeech")
    eval_dataset = CsvDataset(eval_df, tokenizer, input_column="hateSpeech", output_column="counterSpeech")

    # Define a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is not trained with masked language modeling (mlm)
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        gradient_accumulation_steps=4,  # Adjust to fit large models
        report_to="tensorboard",  # Report metrics to TensorBoard
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    main()
