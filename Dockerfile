FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn matplotlib --timeout=100

CMD ["python", "hello_world_ml.py"]
