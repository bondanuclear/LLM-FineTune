import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import AutoTokenizer
from huggingface_hub import interpreter_login

def preprocess_conversation(examples):
    # Конкатенація усіх рядків в один
    conversation = ""
    for message in examples["messages"]:
        role = message["role"].capitalize()
        content = message["content"]
        conversation += f"{role}: {content}\n"
    return {"text": conversation.strip()}
import os
os.environ["WANDB_DISABLED"] = "true"
#interpreter_login()
def train():
    # Завантажуємо датасет
    dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
    val_dataset = load_dataset("HuggingFaceTB/smoltalk","everyday-conversations", split ="test")
    # Попередня обробка датасета - це важливо для тренування
    dataset = dataset.map(preprocess_conversation)
    val_dataset = val_dataset.map(preprocess_conversation)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", device_map="auto")
    tokenizer.chat_template = "<prompt_template>"
    tokenizer.pad_token = tokenizer.eos_token
   
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)  # підготовка моделі для LoRa
   
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
        output_dir="smoltalk-tuned-everyday-10",
        per_device_train_batch_size=8, # визначає, скільки зразків (точок даних) обробляється одночасно на один графічний процесор або пристрій.
        gradient_accumulation_steps=4,  
        optim="adamw_torch",
        logging_steps=10,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        fp16=True, # Дозволяє використовувати 16-бітну точність з плаваючою комою замість 32-бітної точності за замовчуванням.
        warmup_ratio=0.1, # поступово збільшує швидкість навчання для перших 10% кроків навчання.
        lr_scheduler_type="linear",
        num_train_epochs=10, 
        save_strategy="epoch",
        save_total_limit=2, 
        push_to_hub=True,
        hub_model_id="Ridaren/smoltalk-tuned-everyday-10",
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

    # Тренування та збереження
    trainer.train()
    trainer.push_to_hub()

    # Локально зберегти
    model.save_pretrained("smoltalk-everyday-model-10")
    tokenizer.save_pretrained("smoltalk-everyday-tokenizer-10")
    print("finished and saved")

if __name__ == "__main__":
    train()