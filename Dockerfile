FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . .

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
    wget \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.10 \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists


RUN mkdir lora
RUN wget --content-disposition https://civitai.com/api/download/models/1026423  -P lora

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python3.10 -m pip install -r requirements.txt
# RUN python3.10 -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

EXPOSE 8898

CMD python -u app.py 
