FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/facebookresearch/CRAG.git .

COPY requirements.txt ./

RUN pip install --no-cache-dir -r CRAG/requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY run.py .

ENV PYTHONPATH=/workspace

ENTRYPOINT ["python", "run.py"]