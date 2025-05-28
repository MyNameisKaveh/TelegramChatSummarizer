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

# تنظیم cache directory
cache_dir = '/tmp/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# دانلود nltk data
try:
    nltk.download('punkt', download_dir='./nltk_data', quiet=True)
    nltk.data.path.append('./nltk_data')
except:
    pass

# تنظیم logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# حافظه موقتی پیام‌ها برای هر گروه
message_storage = {}
MAX_MESSAGES_PER_CHAT = 1000

# مدل فارسی
MODEL_NAME = "nafisehNik/mt5-persian-summary"
model = None
tokenizer = None

class MessageStore:
    """کلاس برای ذخیره و مدیریت پیام‌ها"""
    
    def __init__(self):
        self.messages = {}
    
    def add_message(self, chat_id: int, user_id: int, username: str, text: str, timestamp: datetime):
        """اضافه کردن پیام جدید"""
        if chat_id not in self.messages:
            self.messages[chat_id] = []
        
        # حفظ حداکثر تعداد پیام
        if len(self.messages[chat_id]) >= MAX_MESSAGES_PER_CHAT:
            self.messages[chat_id] = self.messages[chat_id][-MAX_MESSAGES_PER_CHAT//2:]
        
        self.messages[chat_id].append({
            'user_id': user_id,
            'username': username,
            'text': text,
            'timestamp': timestamp
        })
    
    def get_messages(self, chat_id: int, count: int = 50, hours_back: int = None):
        """دریافت پیام‌ها براساس تعداد یا زمان"""
        if chat_id not in self.messages:
            return []
        
        messages = self.messages[chat_id]
        
        # فیلتر بر اساس زمان
        if hours_back:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            messages = [msg for msg in messages if msg['timestamp'] >= cutoff_time]
        
        # برگرداندن آخرین پیام‌ها
        return messages[-count:] if count else messages

# ایجاد نمونه از مخزن پیام‌ها
message_store = MessageStore()

def load_persian_model():
    """بارگیری مدل فارسی"""
    try:
        logger.info(f"Loading Persian model: {MODEL_NAME}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        model.eval()
        logger.info("Persian model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading Persian model: {e}")
        return None, None

def preprocess_persian_text(text):
    """پیش‌پردازش پیشرفته متن فارسی"""
    # حذف کاراکترهای اضافی و تمیز کردن
    text = re.sub(r'\s+', ' ', text)  # چندین فاصله -> یک فاصله
    text = re.sub(r'\n+', '\n', text)  # چندین خط جدید -> یک خط
    
    # حذف timestamp و نام‌های کاربری تلگرام
    text = re.sub(r'\d{2}:\d{2}', '', text)  # زمان
    text = re.sub(r'@\w+', '', text)  # منشن‌ها
    
    # حذف لینک‌ها
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # حذف ایموجی‌ها
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', ' ', text)
    
    return text.strip()

def chunk_text_smart(text, max_length=300):
    """تقسیم هوشمند متن با در نظر گیری زبان فارسی"""
    try:
        sentences = sent_tokenize(text)
    except:
        # روش جایگزین برای جمله‌بندی فارسی
        sentences = re.split(r'[.!?؟۔]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk + sentence) < max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if len(sentence) < max_length else sentence[:max_length]
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_messages(messages_data):
    """خلاصه‌سازی پیام‌های گروه با مدل فارسی"""
    global model, tokenizer
    
    if not model or not tokenizer:
        return "❌ مدل خلاصه‌سازی در دسترس نیست"
    
    if not messages_data:
        return "❌ پیامی برای خلاصه‌سازی یافت نشد"
    
    try:
        # ترکیب پیام‌ها
        combined_text = ""
        for msg in messages_data:
            user_prefix = f"{msg['username'] or 'کاربر'}: " if msg['username'] else ""
            combined_text += f"{user_prefix}{msg['text']}\n"
        
        # پیش‌پردازش
        combined_text = preprocess_persian_text(combined_text)
        
        if len(combined_text) < 100:
            return "❌ متن برای خلاصه‌سازی بسیار کوتاه است"
        
        # تقسیم به بخش‌های کوچک
        chunks = chunk_text_smart(combined_text, max_length=400)
        summaries = []
        
        for i, chunk in enumerate(chunks[:2]):  # حداکثر 2 بخش
            try:
                inputs = tokenizer.encode(
                    f"خلاصه: {chunk}",
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                
                summary_ids = model.generate(
                    inputs,
                    max_length=100,
                    min_length=30,
                    length_penalty=1.2,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # پاک کردن prefix
                if summary.startswith("خلاصه:"):
                    summary = summary[5:].strip()
                
                summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")
                continue
        
        if not summaries:
            return "❌ خطا در فرآیند خلاصه‌سازی"
        
        # ترکیب نهایی
        final_summary = "\n\n".join(summaries)
        
        # اضافه کردن اطلاعات آماری
        stats = f"\n\n📊 آمار: {len(messages_data)} پیام، {len(combined_text)} کاراکتر"
        
        return f"📝 خلاصه گفتگو:\n\n{final_summary}{stats}"
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"❌ خطا در خلاصه‌سازی: {str(e)}"

def parse_summary_request(text):
    """تجزیه درخواست خلاصه‌سازی"""
    text = text.lower()
    
    # پیدا کردن تعداد پیام
    count_patterns = [
        r'(\d+)\s*پیام',
        r'(\d+)\s*تا',
        r'آخرین\s*(\d+)',
    ]
    
    message_count = 50  # پیش‌فرض
    
    for pattern in count_patterns:
        match = re.search(pattern, text)
        if match:
            message_count = min(int(match.group(1)), 200)  # حداکثر 200
            break
    
    # پیدا کردن بازه زمانی
    time_patterns = [
        r'(\d+)\s*ساعت',
        r'(\d+)\s*روز',
    ]
    
    hours_back = None
    
    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            if 'روز' in pattern:
                hours_back = int(match.group(1)) * 24
            else:
                hours_back = int(match.group(1))
            hours_back = min(hours_back, 72)  # حداکثر 3 روز
            break
    
    return message_count, hours_back

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """شروع ربات"""
    welcome_msg = f"""
🤖 سلام! من ربات خلاصه‌ساز گروه هستم.

📋 برای استفاده از من:
- من را با @{context.bot.username} تگ کنید
- بعد عبارت "خلاصه" یا "خلاصه کن" بنویسید
    
🔹 مثال‌ها:
• @{context.bot.username} خلاصه کن
• @{context.bot.username} خلاصه 100 پیام آخر
• @{context.bot.username} خلاصه 2 ساعت اخیر

⚙️ دستورات:
/help - راهنمای کامل
/stats - آمار گروه
/model - اطلاعات مدل فعلی

🔸 توجه: من فقط وقتی تگ شوم کار می‌کنم!
    """
    
    await update.message.reply_text(welcome_msg)

async def model_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """نمایش اطلاعات مدل فعلی"""
    info_text = f"""
🤖 اطلاعات مدل فعلی:

📦 نام مدل: {MODEL_NAME}
🌐 پشتیبانی زبان: ✅ فارسی
💾 وضعیت: فعال و آماده
    """
    await update.message.reply_text(info_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """راهنمای کامل"""
    help_text = f"""
🤖 راهنمای ربات خلاصه‌ساز

📝 نحوه استفاده:
1. من را با @{context.bot.username} تگ کنید
2. کلمه "خلاصه" یا "خلاصه کن" اضافه کنید
3. اختیاری: تعداد پیام یا بازه زمانی مشخص کنید

🔹 مثال‌های مختلف:
• @{context.bot.username} خلاصه کن
• @{context.bot.username} خلاصه 50 پیام
• @{context.bot.username} خلاصه 3 ساعت اخیر

⚡ ویژگی‌ها:
• پردازش تا 200 پیام
• بازه زمانی تا 3 روز
• پشتیبانی از متن فارسی
• تطبیق خودکار با بهترین مدل موجود

🔧 دستورات:
/start - شروع
/help - راهنما
/stats - آمار گروه
/model - اطلاعات مدل

🔸 نکته: من فقط در گروه‌ها و وقتی تگ شوم کار می‌کنم!
    """
    await update.message.reply_text(help_text)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """نمایش آمار گروه"""
    chat_id = update.effective_chat.id
    
    if chat_id not in message_store.messages:
        await update.message.reply_text("📊 هنوز پیامی ذخیره نشده")
        return
    
    messages = message_store.messages[chat_id]
    total_messages = len(messages)
    
    # شمارش پیام‌ها در 24 ساعت اخیر
    day_ago = datetime.now() - timedelta(hours=24)
    recent_messages = len([m for m in messages if m['timestamp'] >= day_ago])
    
    # کاربران فعال
    users = {}
    for msg in messages:
        username = msg['username'] or 'کاربر ناشناس'
        users[username] = users.get(username, 0) + 1
    
    top_users = sorted(users.items(), key=lambda x: x[1], reverse=True)[:5]
    
    stats_text = f"""
📊 آمار گروه:

📈 کل پیام‌های ذخیره شده: {total_messages}
🕐 پیام‌های 24 ساعت اخیر: {recent_messages}

👥 کاربران فعال:
"""
    
    for username, count in top_users:
        stats_text += f"• {username}: {count} پیام\n"
    
    await update.message.reply_text(stats_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ذخیره پیام‌ها و پردازش درخواست‌های خلاصه"""
    
    # اگر پیام خصوصی باشد، نادیده بگیر
    if update.effective_chat.type == 'private':
        return
    
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    username = update.effective_user.username
    message_text = update.message.text
    timestamp = datetime.now()
    
    # ذخیره پیام (همیشه)
    message_store.add_message(chat_id, user_id, username, message_text, timestamp)
    
    # بررسی اینکه آیا ربات تگ شده یا نه
    if not context.bot.username:
        return
        
    bot_mention = f"@{context.bot.username.lower()}"
    message_lower = message_text.lower()
    
    # اگر ربات تگ نشده، کاری نکن
    if bot_mention not in message_lower:
        return
    
    # بررسی درخواست خلاصه
    summary_keywords = ['خلاصه', 'خلاصه کن', 'summarize', 'خلاصه بده']
    
    if not any(keyword in message_lower for keyword in summary_keywords):
        return
    
    # ارسال پیام "در حال پردازش"
    processing_msg = await update.message.reply_text("⏳ در حال جمع‌آوری و خلاصه‌سازی پیام‌ها...")
    
    try:
        # تجزیه درخواست
        message_count, hours_back = parse_summary_request(message_text)
        
        # دریافت پیام‌ها
        messages_data = message_store.get_messages(chat_id, message_count, hours_back)
        
        if not messages_data:
            await processing_msg.edit_text("❌ پیامی برای خلاصه‌سازی یافت نشد")
            return
        
        # خلاصه‌سازی
        summary = summarize_messages(messages_data)
        
        # ارسال نتیجه
        await processing_msg.edit_text(summary)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await processing_msg.edit_text(f"❌ خطا در پردازش: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """مدیریت خطاها"""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """تابع اصلی"""
    global model, tokenizer
    
    # دریافت توکن
    BOT_TOKEN = os.environ.get('BOT_TOKEN')
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not found!")
        return
    
    # بارگیری مدل فارسی
    logger.info("Loading Persian model...")
    model, tokenizer = load_persian_model()
    
    if not model:
        logger.error("Failed to load any model!")
        return
    
    # ساخت اپلیکیشن
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # اضافه کردن handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("model", model_info))
    
    # Handler برای تمام پیام‌ها (ذخیره + پردازش)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    # شروع
    logger.info(f"Bot started with Persian model: {MODEL_NAME}")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
