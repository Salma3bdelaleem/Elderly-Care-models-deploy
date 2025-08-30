# الصورة الأساسية
FROM python:3.10-slim

# تثبيت المتطلبات الأساسية للنظام
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد للتطبيق
WORKDIR /app

# نسخ الملفات
COPY . /app

# تثبيت الباكيجات
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# فتح البورت
EXPOSE 8000

# الأمر الأساسي لتشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
