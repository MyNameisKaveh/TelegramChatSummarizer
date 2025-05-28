import os
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Bot

# تنظیم مسیر کش (برای Spaces مناسب)
os.environ["TRANSFORMERS_CACHE"] = "/data/.cache/huggingface"

MODEL_NAME = "nafisehNik/mt5-persian-summary"

def get_summarizer_model():
    print("[LOG] Loading model from:", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("[LOG] Model loaded.")
    
    print("[LOG] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("[LOG] Tokenizer loaded.")
    
    return model, tokenizer

async def startup():
    print("[LOG] Inside async startup...")
    model, tokenizer = get_summarizer_model()
    print("[LOG] Model and tokenizer loaded in startup().")

    # برای تست اولیه بدون ارسال پیام، فقط یک لاگ بذار
    print("[LOG] Bot is ready.")

    # اگر خواستی پیام بفرسته، این قسمت رو فعال کن
    # bot = Bot(token="YOUR_TELEGRAM_BOT_TOKEN")
    # await bot.send_message(chat_id="YOUR_CHAT_ID", text="Bot is ready.")

if __name__ == "__main__":
    print("===== Application Startup =====")
    scheduler = BackgroundScheduler()
    scheduler.start()
    print("[LOG] Scheduler started.")

    loop = asyncio.get_event_loop()
    print("[LOG] Running startup coroutine...")
    loop.run_until_complete(startup())
    print("[LOG] Startup complete.")
