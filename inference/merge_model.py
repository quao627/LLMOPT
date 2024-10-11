import argparse
import json
import os
from threading import Thread
import sys
from copy import deepcopy
from transformers.trainer_utils import set_seed
import platform
import torch
from peft import PeftModel
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import transformers
from transformers.generation import GenerationConfig
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)


tokenizer_pth = ""
model_pth = ""
base_pth = ""
saved_pth = ""


MODEL_CLASSES = {
    "copilot": {AutoPeftModelForCausalLM, AutoTokenizer},
    "qwen": {AutoModelForCausalLM, AutoTokenizer},
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

def load_model(tokenizer_pth, model_pth):
    model_class, tokenizer_class = MODEL_CLASSES['copilot']
    tokenizer = tokenizer_class.from_pretrained(tokenizer_pth, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_pth,
        device_map="auto",
        trust_remote_code=True
    )
    config = GenerationConfig.from_pretrained(
        model_pth, trust_remote_code=True
    )
    return model, tokenizer, config

def merge_model_from_dir(base_model_path, peft_model_path):
    print(f"Starting to load the model {base_model_path} into memory")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, peft_model_path)
    model = model.merge_and_unload()
    return model


model = merge_model_from_dir(base_pth, model_pth)
print("model saving!")
model.save_pretrained(saved_pth)
print("model saved!")
