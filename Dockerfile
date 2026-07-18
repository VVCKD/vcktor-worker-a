# VCKTOR Worker A — RunPod Serverless (커스텀 pth 모델 전용 RVC 변환)
FROM nvidia/cuda:12.3.0-base-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    wget unzip ffmpeg libsndfile1 aria2 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip==24.0
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p assets/hubert assets/rmvpe assets/weights assets/indices assets/uvr5_weights logs

# 추론용 베이스 모델만 (hubert=특성추출, rmvpe=f0)
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
    -d assets/hubert -o hubert_base.pt
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt \
    -d assets/rmvpe -o rmvpe.pt

COPY . .

ENV weight_root=/workspace/assets/weights \
    index_root=/workspace/assets/indices \
    outside_index_root=/workspace/assets/indices \
    rmvpe_root=/workspace/assets/rmvpe \
    weight_uvr5_root=/workspace/assets/uvr5_weights \
    log_root=/workspace/logs \
    PYTHONUNBUFFERED=1

CMD ["python3", "-u", "handler.py"]
