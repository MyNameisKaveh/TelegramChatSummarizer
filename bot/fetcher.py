from telegram.ext import CallbackContext
from datetime import datetime
from typing import Optional

# در حال حاضر فقط پیام‌هایی که خود ربات دریافت کرده رو نگه می‌داریم
# در مرحله بعدی میشه از database یا cache استفاده کرد

# حافظه موقتی پیام‌ها
message_log = {}

def save_message(chat_id: int, text: str, timestamp: datetime):
    if chat_id not in message_log:
        message_log[chat_id] = []
    message_log[chat_id].append((text, timestamp))

def fetch_messages(chat_id: int, count: int = 50, since: Optional[datetime] = None):
    if chat_id not in message_log:
        return []

    messages = message_log[chat_id]

    if since:
        filtered = [text for text, ts in messages if ts >= since]
    else:
        filtered = [text for text, _ in messages]

    return filtered[-count:]
