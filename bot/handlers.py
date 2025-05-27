from telegram import Update
from telegram.ext import CallbackContext
from bot.fetcher import fetch_messages
from bot.responder import send_summary
from summarizer.summarize import summarize_text
import re
from datetime import datetime, timedelta

def handle_message(update: Update, context: CallbackContext):
    message_text = update.message.text.lower()
    chat_id = update.effective_chat.id

    # بررسی اینکه ربات تگ شده یا نه
    if context.bot.username.lower() not in message_text:
        return

    # استخراج تعداد پیام یا بازه زمانی
    num_messages = 50  # مقدار پیش‌فرض
    time_limit = None

    # بررسی عدد در پیام (مثلاً "100 پیام آخر")
    count_match = re.search(r'(\d+)\s*(پیام|تا پیام)', message_text)
    if count_match:
        num_messages = int(count_match.group(1))

    # بررسی بازه زمانی (مثلاً "در 2 ساعت اخیر")
    time_match = re.search(r'(\d+)\s*ساعت', message_text)
    if time_match:
        hours = int(time_match.group(1))
        time_limit = datetime.now() - timedelta(hours=hours)

    # گرفتن پیام‌ها
    messages = fetch_messages(chat_id, num_messages, time_limit)

    if not messages:
        context.bot.send_message(chat_id=chat_id, text="هیچ پیامی برای خلاصه‌سازی پیدا نشد.")
        return

    # ساخت متن کامل برای خلاصه
    full_text = "\n".join(messages)
    summary = summarize_text(full_text)

    # ارسال خلاصه
    send_summary(context.bot, chat_id, summary)
