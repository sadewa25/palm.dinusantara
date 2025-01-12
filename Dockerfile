FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev python3-dev libc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --upgrade pip \
    && pip install "pybind11>=2.12" opencv-python-headless "numpy>=2.0" 

RUN pip install -r requirements_api.txt 
    # && pip install --no-deps nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl \
    # && pip install --no-deps nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl \
    # && pip install --no-deps nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl \
    # && pip install --no-deps nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl


FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY model ./model/
COPY static ./static/
COPY ext.py .
COPY api.py .



# COPY utils ./utils/
# COPY .env .
# COPY app.py .
# COPY ext.py .
# COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]