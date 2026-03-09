FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY serve.py .

EXPOSE 5002

CMD ["python", "serve.py"]
