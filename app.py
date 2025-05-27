import logging
from telegram.ext import Updater, MessageHandler, Filters
from bot.handlers import handle_message
from bot.fetcher import save_message
import os
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

TOKEN = os.environ.get("BOT_TOKEN")

def save_incoming_message(update, context):
    chat_id = update.effective_chat.id
    message = update.message.text
    timestamp = datetime.fromtimestamp(update.message.date.timestamp())

    if message:
        save_message(chat_id, message, timestamp)

def main():
    if not TOKEN:
        logger.error("توکن ربات تنظیم نشده. متغیر محیطی BOT_TOKEN را تنظیم کن.")
        return

    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    # ذخیره‌سازی هر پیام متنی که میاد
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), save_incoming_message), group=0)

    # هندل پیام‌های فرمان خلاصه
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_message), group=1)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
