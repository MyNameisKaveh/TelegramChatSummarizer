import os
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk
from nltk.tokenize import sent_tokenize
import torch

# تنظیم مسیر cache برای Transformers
#cache_dir = '/tmp/transformers_cache'
#os.environ['TRANSFORMERS_CACHE'] = cache_dir
#os.environ['HF_HOME'] = cache_dir
#os.makedirs(cache_dir, exist_ok=True)

# تنظیم مسیر nltk
try:
    nltk.download('punkt', download_dir='./nltk_data', quiet=True)
    nltk.data.path.append('./nltk_data')
except:
    pass

# تنظیمات لاگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# اطلاعات مدل
MODEL_NAME = "nafisehNik/mt5-persian-summary"
model = None
tokenizer = None

# ذخیره پیام‌ها برای هر چت
MAX_MESSAGES_PER_CHAT = 1000
class MessageStore:
    def __init__(self):
        self.messages = {}

    def add_message(self, chat_id, user_id, username, text, timestamp):
        if chat_id not in self.messages:
            self.messages[chat_id] = []

        if len(self.messages[chat_id]) >= MAX_MESSAGES_PER_CHAT:
            self.messages[chat_id] = self.messages[chat_id][-MAX_MESSAGES_PER_CHAT // 2:]

        self.messages[chat_id].append({
            "user_id": user_id,
            "username": username,
            "text": text,
            "timestamp": timestamp
        })

    def get_messages(self, chat_id, count=50, hours_back=None):
        if chat_id not in self.messages:
            return []

        messages = self.messages[chat_id]

        if hours_back:
            cutoff = datetime.now() - timedelta(hours=hours_back)
            messages = [m for m in messages if m["timestamp"] >= cutoff]

        return messages[-count:] if count else messages

message_store = MessageStore()

def load_persian_model():
    global model, tokenizer
    try:
        logger.info(f"Loading Persian model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Persian model: {e}")
        model, tokenizer = None, None

def preprocess_persian_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\d{2}:\d{2}', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return text.strip()

def chunk_text_smart(text, max_length=300):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = re.split(r'[.!?؟]+', text)

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current + sentence) < max_length:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks

def summarize_messages(messages_data):
    global model, tokenizer
    if not model or not tokenizer:
        return "❌ مدل خلاصه‌سازی در دسترس نیست"
    if not messages_data:
        return "❌ پیامی برای خلاصه‌سازی یافت نشد"

    try:
        text = ""
        for msg in messages_data:
            username = msg['username'] or "کاربر"
            text += f"{username}: {msg['text']}\n"

        text = preprocess_persian_text(text)
        if len(text) < 100:
            return "❌ متن برای خلاصه‌سازی بسیار کوتاه است"

        chunks = chunk_text_smart(text, max_length=400)
        summaries = []

        for chunk in chunks[:2]:
            inputs = tokenizer.encode(f"خلاصه: {chunk}", return_tensors="pt", max_length=512, truncation=True)
            output = model.generate(
                inputs,
                max_length=100,
                min_length=30,
                length_penalty=1.2,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary.replace("خلاصه:", "").strip())

        if not summaries:
            return "❌ خطا در خلاصه‌سازی"

        stats = f"\n\n📊 آمار: {len(messages_data)} پیام، {len(text)} کاراکتر"
        return f"📝 خلاصه گفتگو:\n\n" + "\n\n".join(summaries) + stats

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return "❌ خطا در خلاصه‌سازی"

def parse_summary_request(text):
    text = text.lower()
    count = 50
    hours = None

    match = re.search(r'(\d+)\s*(پیام|تا|عدد)', text)
    if match:
        count = min(int(match.group(1)), 200)

    match = re.search(r'(\d+)\s*(ساعت|روز)', text)
    if match:
        hours = int(match.group(1))
        if "روز" in match.group(2):
            hours *= 24
        hours = min(hours, 72)

    return count, hours

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 سلام! برای خلاصه‌سازی، عبارت «خلاصه» به همراه تعداد پیام یا مدت زمان را بفرست.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    user_id = message.from_user.id
    username = message.from_user.username
    text = message.text.strip()
    timestamp = message.date or datetime.utcnow()

    message_store.add_message(chat_id, user_id, username, text, timestamp)

    if "خلاصه" in text:
        count, hours = parse_summary_request(text)
        msgs = message_store.get_messages(chat_id, count, hours)
        summary = summarize_messages(msgs)
        await update.message.reply_text(summary)

if __name__ == "__main__":
    load_persian_model()
    TOKEN = os.getenv("BOT_TOKEN")  # یا مستقیم وارد کن: 'your_token_here'

    if not TOKEN:
        raise ValueError("❌ توکن تلگرام تعریف نشده.")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting bot...")
    app.run_polling()
