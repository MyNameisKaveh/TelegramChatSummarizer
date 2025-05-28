import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    model_name = "nafisehNik/mt5-persian-summary"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # بدون cache_dir
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}
