from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def load_model():
    model_name = "nafisehNik/mt5-persian-summary"
    cache_dir = "/tmp/hf_cache"

    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    return tokenizer, model
