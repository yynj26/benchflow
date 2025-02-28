FROM python:3.10-slim

WORKDIR /app

#prepare git if necessary for clone repository
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir benchflow litellm pydantic

RUN git clone https://github.com/sierra-inc/taubench.git
WORKDIR /app/taubench
RUN pip install -e .
WORKDIR /app


RUN mkdir -p /app/results /app/log_files

COPY entrypoint.py /app/

ENV PYTHONPATH="${PYTHONPATH}:/app/taubench"
ENV PYTHONUNBUFFERED=1

ENV ENV=retail
ENV USER_STRATEGY=llm
ENV USER_MODEL=gpt-4o
ENV USER_MODEL_PROVIDER=openai
ENV TASK_SPLIT=test
ENV LOG_DIR=/app/results
ENV MAX_STEPS=30

ENTRYPOINT ["python", "/app/entrypoint.py"]