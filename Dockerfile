FROM python:3.12.7

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt .
COPY setup.py .
COPY src/ src/

RUN uv pip install -r requirements.txt
RUN uv pip install -e .

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "src.ocr_bank_docs.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 
