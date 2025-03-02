import json
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

# -------------------------------
# 1. Custom Dataset with Persona
# -------------------------------
class PersonaChatDataset(Dataset):
    def __init__(self, json_file, tokenizer, persona, max_length=256):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.persona = persona
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        # Build text with persona, user prompt, and Emet-Selch response
        text = (
            f"{self.persona}\n"  # Persona line
            f"User: {pair['input']}\n"
            f"Emet-Selch: {pair['output']}"
        )

        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove batch dimension
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

def main():
    # 2. Define persona text
    persona_text = "<|persona|> Emet-Selch is condescending, sarcastic, and mocks mortals at every opportunity."

    # 3. Load GPT-2 Large
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token by default; set to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Create dataset
    dataset = PersonaChatDataset(
        json_file="filtered_dialogue_pairs.json",
        tokenizer=tokenizer,
        persona=persona_text,
        max_length=256
    )
    print("Dataset size:", len(dataset))

    # 5. Load the GPT-2 Large model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 6. Set training arguments
    training_args = TrainingArguments(
        output_dir="./emetselch_gpt2large",
        num_train_epochs=3,             # Adjust as needed
        per_device_train_batch_size=1,  # GPT-2 Large can be memory-heavy, try batch_size=1 or 2
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        # If you have a validation set, you can add evaluation_strategy="steps" or "epoch"
    )

    # 7. Create trainer & train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # 8. Save final model
    model.save_pretrained("./emetselch_gpt2large")
    tokenizer.save_pretrained("./emetselch_gpt2large")
    print("Model saved to './emetselch_gpt2large'")

if __name__ == "__main__":
    main()