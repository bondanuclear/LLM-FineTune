# Use a pipeline as a high-level helper
from transformers import pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
#generator = pipeline("text-generation", model="Ridaren/smoltalk-tuned-everyday-10")
generator = pipeline("text-generation", model="Ridaren/gpt2-socratic")

while True:
    input_text = input("Your line: ")

    if input_text.lower() == 'quit':
        break

    output = generator(input_text, max_new_tokens=512, return_full_text=True)
    
    generated_text = output[0]["generated_text"]

    print(generated_text)

