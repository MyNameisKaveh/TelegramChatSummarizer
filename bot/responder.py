from telegram import Bot

def send_summary(bot: Bot, chat_id: int, summary: str):
    # می‌تونی اینجا بعداً قالب‌بندی یا شکل‌های بهتری اضافه کنی
    bot.send_message(chat_id=chat_id, text=summary)
