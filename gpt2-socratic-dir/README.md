---
base_model: distilbert/distilgpt2
library_name: transformers
model_name: gpt2-socratic
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for gpt2-socratic

This model is a fine-tuned version of [distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="Ridaren/gpt2-socratic", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure



This model was trained with SFT.

### Framework versions

- TRL: 0.12.1
- Transformers: 4.46.3
- Pytorch: 2.5.1
- Datasets: 3.1.0
- Tokenizers: 0.20.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin GallouГ©dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```