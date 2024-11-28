import numpy
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import AutoTokenizer
arr = numpy.array([1, 2, 3, 4, 5])

print(arr)