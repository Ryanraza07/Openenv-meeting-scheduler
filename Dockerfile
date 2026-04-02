FROM python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
ENV PYTHONPATH=/app