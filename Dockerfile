FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling huge CUDA wheels in Linux containers.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

RUN mkdir -p vectordb documents evaluation

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
