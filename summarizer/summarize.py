from summarizer.models import load_model

# مدل رو فقط یک‌بار بارگذاری می‌کنیم
model = load_model()

def summarize_text(text: str) -> str:
    if not text.strip():
        return "متنی برای خلاصه‌سازی وجود ندارد."

    # تقسیم به پاراگراف‌های کوچکتر اگه خیلی بزرگ باشه
    if len(text) > 2000:
        text = text[:2000]

    try:
        return model(text)
    except Exception as e:
        return f"در خلاصه‌سازی خطا رخ داد: {str(e)}"
