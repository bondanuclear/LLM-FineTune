import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import AutoTokenizer
from huggingface_hub import interpreter_login

import os
os.environ["WANDB_DISABLED"] = "true"



def preprocess_dataset(dataset):
    dataset = dataset.map(lambda x: {"text": x["question"] + " " + x["answer"]})
    return dataset



def train():

    dataset = load_dataset("openai/gsm8k", "socratic", split="train")
    val_dataset =load_dataset("openai/gsm8k", "socratic", split ="test")

    dataset = preprocess_dataset(dataset)
    val_dataset = preprocess_dataset(val_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", device_map = "auto")
    
    tokenizer.chat_template = "<prompt_template>"
    tokenizer.pad_token = tokenizer.eos_token
   
    model.resize_token_embeddings(len(tokenizer))  
    model = prepare_model_for_kbit_training(model)  # Prepare the model for LoRA
   
    # Configure LoRA
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)


    # Training arguments
    training_args = TrainingArguments(
        output_dir="gpt2-socratic-dir",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2, 
        optim="adamw_torch",
        logging_steps=10,
        evaluation_strategy="epoch",  
        learning_rate=5e-5,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=2, 
        save_strategy="epoch",
        save_total_limit=2,  
        push_to_hub=True,
        hub_model_id="Ridaren/gpt2-socratic", 
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset, 
        dataset_text_field="text", 
        max_seq_length=256,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,  # 
        peft_config=peft_config,
    )

    # Train and save
    trainer.train()
    trainer.push_to_hub()

    # Save locally
    model.save_pretrained("distilgpt2-socratic")
    tokenizer.save_pretrained("distilgpt2-socratic")
    print("finished and saved")

if __name__ == "__main__":
    train()