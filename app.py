import os
import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
from nltk.tokenize import sent_tokenize

# تنظیم cache directory برای transformers - استفاده از /tmp
cache_dir = '/tmp/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

# ایجاد دایرکتوری cache اگر وجود نداره
os.makedirs(cache_dir, exist_ok=True)

# دانلود nltk data
try:
    nltk.download('punkt', download_dir='./nltk_data')
    nltk.data.path.append('./nltk_data')
except:
    pass

# تنظیم logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# متغیرهای مدل - استفاده از مدل کوچکتر
MODEL_NAME = "facebook/bart-large-cnn"  # مدل کوچکتر و سریعتر
model = None
tokenizer = None

def get_summarizer_model():
    """بارگیری مدل و tokenizer"""
    try:
        cache_dir = '/tmp/transformers_cache'
        # استفاده از cache directory سفارشی
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False,
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # اگر مدل اصلی کار نکرد، از مدل جایگزین استفاده کن
        try:
            logger.info("Trying alternative model...")
            alt_model_name = "sshleifer/distilbart-cnn-12-6"
            model = AutoModelForSeq2SeqLM.from_pretrained(
                alt_model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            tokenizer = AutoTokenizer.from_pretrained(
                alt_model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            return model, tokenizer
        except Exception as e2:
            logger.error(f"Error loading alternative model: {e2}")
            return None, None

def preprocess_text(text):
    """پیش‌پردازش متن"""
    # حذف کاراکترهای اضافی
    text = text.replace('\n', ' ').replace('\r', ' ')
    # حذف فاصله‌های اضافی
    text = ' '.join(text.split())
    return text

def chunk_text(text, max_length=512):
    """تقسیم متن به قطعات کوچکتر"""
    try:
        sentences = sent_tokenize(text)
    except:
        # اگر nltk کار نکرد، به صورت ساده تقسیم می‌کنیم
        sentences = text.split('. ')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) < max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_text(text):
    """خلاصه‌سازی متن"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "مدل بارگیری نشده است."
    
    try:
        # پیش‌پردازش
        text = preprocess_text(text)
        
        if len(text) < 100:
            return "متن برای خلاصه‌سازی بسیار کوتاه است."
        
        # تقسیم به قطعات
        chunks = chunk_text(text, max_length=400)
        summaries = []
        
        for chunk in chunks:
            inputs = tokenizer.encode(
                "summarize: " + chunk,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            summary_ids = model.generate(
                inputs,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        
        # ترکیب خلاصه‌ها
        final_summary = " ".join(summaries)
        return final_summary
        
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"خطا در خلاصه‌سازی: {str(e)}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """شروع بات"""
    await update.message.reply_text(
        "سلام! من ربات خلاصه‌ساز هستم.\n"
        "متن خود را برای من ارسال کنید تا آن را خلاصه کنم.\n"
        "برای راهنمایی /help را ارسال کنید."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """راهنمای استفاده"""
    help_text = """
🤖 راهنمای استفاده از ربات خلاصه‌ساز:

📝 برای خلاصه‌سازی:
- متن خود را مستقیماً ارسال کنید
- متن باید حداقل ۱۰۰ کاراکتر باشد

⚡ نکات مهم:
- متن‌های طولانی ممکن است زمان بیشتری نیاز داشته باشند
- کیفیت خلاصه بستگی به محتوای متن دارد

📧 دستورات:
/start - شروع مجدد
/help - راهنمای استفاده
    """
    await update.message.reply_text(help_text)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """پردازش متن‌های ارسالی"""
    user_text = update.message.text
    
    if not user_text or len(user_text.strip()) < 50:
        await update.message.reply_text("لطفاً متن طولانی‌تری ارسال کنید (حداقل ۵۰ کاراکتر)")
        return
    
    # ارسال پیام "در حال پردازش"
    processing_message = await update.message.reply_text("⏳ در حال خلاصه‌سازی...")
    
    try:
        # خلاصه‌سازی
        summary = summarize_text(user_text)
        
        # ارسال نتیجه
        result_text = f"📝 خلاصه متن:\n\n{summary}"
        
        await processing_message.edit_text(result_text)
        
    except Exception as e:
        logger.error(f"Error handling text: {e}")
        await processing_message.edit_text(f"❌ خطا در پردازش: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """مدیریت خطاها"""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """تابع اصلی"""
    global model, tokenizer
    
    # دریافت توکن از متغیر محیطی
    BOT_TOKEN = os.environ.get('BOT_TOKEN')
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not found in environment variables")
        return
    
    # بارگیری مدل
    logger.info("Loading model...")
    model, tokenizer = get_summarizer_model()
    
    if model is None:
        logger.error("Failed to load model")
        return
    
    logger.info("Model loaded successfully")
    
    # ساخت اپلیکیشن
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # اضافه کردن handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(error_handler)
    
    # شروع polling
    logger.info("Starting bot...")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
