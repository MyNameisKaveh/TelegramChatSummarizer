---
title: Telegram Chat Summarizer
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860 # Port your app's health check server listens on
pinned: false 
---
# Telegram Chat Summarizer Bot

## توضیحات پروژه (Project Description)

این پروژه یک ربات تلگرامی است که به منظور خلاصه‌سازی گفتگوهای اخیر در گروه‌های تلگرامی طراحی شده است. ربات با دریافت دستوری خاص، پیام‌های اخیر را جمع‌آوری کرده و خلاصه‌ای از آن‌ها را به شما ارائه می‌دهد. این ربات برای زبان فارسی بهینه‌سازی شده است.

## قابلیت‌ها (Features)

*   خلاصه‌سازی پیام‌های اخیر در یک گروه تلگرامی.
*   قابلیت تعیین تعداد پیام‌های مورد نظر برای خلاصه‌سازی (مثلاً ۱۰۰ پیام آخر).
*   قابلیت تعیین بازه زمانی برای جمع‌آوری پیام‌ها (مثلاً پیام‌های ۲۴ ساعت گذشته).
*   پاسخ به دستور "خلاصه" در صورتی که ربات در گروه تگ شود.
*   استفاده از مدل‌های به‌روز یادگیری ماشین برای خلاصه‌سازی به زبان فارسی.

## تکنولوژی‌های استفاده شده (Technologies Used)

*   **زبان برنامه‌نویسی:** پایتون نسخه ۳.۱۰ (Python 3.10)
*   **کتابخانه اصلی ربات تلگرام:** [`python-telegram-bot`](https://python-telegram-bot.org/)
*   **مدل‌های زبانی و پردازش متن:**
    *   کتابخانه [`transformers`](https://huggingface.co/docs/transformers/index) از Hugging Face
    *   کتابخانه [`torch`](https://pytorch.org/) (PyTorch)
    *   کتابخانه [`nltk`](https://www.nltk.org/) (Natural Language Toolkit) برای توکن‌بندی جملات.
*   **مدل خلاصه‌سازی پیش‌فرض:** [`nafisehNik/mt5-persian-summary`](https://huggingface.co/nafisehNik/mt5-persian-summary)

## راه‌اندازی و پیکربندی (Setup & Configuration)

### پیش‌نیازها
*   پایتون نسخه ۳.۱۰ یا بالاتر.
*   ابزار `pip` برای نصب وابستگی‌ها.

### مراحل نصب
1.  **کلون کردن ریپازیتوری:**
    ```bash
    git clone <URL_ریپازیتوری_شما>
    cd <نام_پوشه_پروژه>
    ```

2.  **نصب وابستگی‌ها:**
    یک محیط مجازی پایتون ایجاد و فعال کنید (اختیاری اما به‌شدت توصیه می‌شود):
    ```bash
    python -m venv venv
    source venv/bin/activate  # در لینوکس و macOS
    # venv\Scripts ctivate    # در ویندوز
    ```
    سپس وابستگی‌ها را نصب کنید:
    ```bash
    pip install -r requirements.txt
    ```

3.  **تنظیم توکن ربات تلگرام:**
    ربات برای اتصال به API تلگرام نیاز به یک توکن دارد. این توکن را از BotFather در تلگرام دریافت کنید.
    سپس، توکن را به عنوان یک متغیر محیطی (Environment Variable) با نام `BOT_TOKEN` تنظیم کنید.
    ```bash
    export BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
    ```
    یا می‌توانید آن را مستقیماً در فایل `app.py` قرار دهید (که البته از نظر امنیتی توصیه نمی‌شود).

## اجرای ربات

### اجرای محلی (Running Locally)
پس از نصب وابستگی‌ها و تنظیم توکن، می‌توانید ربات را به صورت محلی اجرا کنید:
```bash
python app.py
```
لاگ‌های مربوط به بارگذاری مدل و شروع به کار ربات در ترمینال نمایش داده خواهند شد.

اجرای با Docker (Docker Deployment)
این پروژه شامل یک Dockerfile است که امکان کانتینری کردن برنامه و اجرای آن در محیط‌هایی مانند Hugging Face Spaces را فراهم می‌کند.

ساخت ایمیج داکر:
docker build -t telegram-summarizer-bot .
اجرای کانتینر داکر:
docker run -e BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN" telegram-summarizer-bot
تنظیمات کش در Dockerfile: Dockerfile به گونه‌ای تنظیم شده است که پوشه‌های کش برای مدل‌های Hugging Face و داده‌های NLTK را در مسیرهای مناسب داخل کانتینر ایجاد و تنظیم کند:

ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV NLTK_DATA=/app/nltk_data این تنظیمات برای سازگاری بهتر با پلتفرم‌هایی مانند Hugging Face Spaces انجام شده است.
نحوه استفاده (Usage)
ربات را به گروه تلگرامی مورد نظر خود اضافه کنید.
برای درخواست خلاصه، پیامی حاوی کلمه "خلاصه" ارسال کنید و ربات را در آن پیام تگ (mention) کنید. مثال‌ها:
@نام_کاربری_ربات خلاصه (خلاصه‌سازی با تنظیمات پیش‌فرض، معمولاً حدود ۵۰ پیام آخر)
@نام_کاربری_ربات خلاصه ۱۰۰ پیام (خلاصه‌سازی ۱۰۰ پیام آخر)
@نام_کاربری_ربات خلاصه در ۲ ساعت اخیر (خلاصه‌سازی پیام‌های ۲ ساعت گذشته)
@نام_کاربری_ربات خلاصه ۱۰۰ پیام در ۳ ساعت اخیر
ربات پیام‌های مشخص شده را جمع‌آوری کرده و خلاصه‌ای از آن‌ها را در گروه ارسال می‌کند.

مدل خلاصه‌سازی (Summarization Model)
مدل پیش‌فرض استفاده شده برای خلاصه‌سازی nafisehNik/mt5-persian-summary است. در صورت تمایل، می‌توانید این مدل را با ویرایش متغیر MODEL_NAME در فایل app.py تغییر دهید. مطمئن شوید مدلی که انتخاب می‌کنید با AutoModelForSeq2SeqLM از کتابخانه transformers سازگار باشد.
