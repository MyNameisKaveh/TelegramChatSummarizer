import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    model_name = "nafisehNik/mt5-persian-summary"
    
    cache_dir = "./cache"  # مطمئن شو این مسیر توی ریپو هست

    os.makedirs(cache_dir, exist_ok=True)  # فولدر بساز اگه نبود

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

    return model, tokenizer
