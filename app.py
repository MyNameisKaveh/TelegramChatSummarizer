import logging
from telegram.ext import Updater, MessageHandler, Filters
from bot.handlers import handle_message
import os

# لاگ برای دیباگ راحت‌تر
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# گرفتن توکن از متغیر محیطی (امن‌تر)
TOKEN = os.environ.get("BOT_TOKEN")

def main():
    if not TOKEN:
        logger.error("توکن ربات تنظیم نشده. متغیر محیطی BOT_TOKEN را تنظیم کن.")
        return

    # ساخت ربات
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    # گرفتن همه پیام‌ها و فرستادنشون به handler
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_message))

    # شروع ربات
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
