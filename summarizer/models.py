from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_model():
    model_name = "m3hrdadfi/bert2bert-fa-summarizer"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    def summarize(text):
        result = summarizer(text, max_length=128, min_length=30, do_sample=False)
        return result[0]["summary_text"]

    return summarize
