# استفاده از ایمیج پایه پایتون
FROM python:3.10-slim

# پوشه کاری در کانتینر
WORKDIR /app

# کپی کردن فایل requirements
COPY requirements.txt .

# نصب وابستگی‌ها
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کل کد پروژه
COPY . .

# فرمان اجرا (اگر اسکریپت اصلی app.py است)
CMD ["python", "app.py"]
