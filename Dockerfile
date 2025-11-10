# DeepSeek-OCR service on DGX Spark (GB10 / Blackwell)
# Base: NVIDIA NGC PyTorch container (25.10-py3)
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Basic env + cache locations
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    TORCH_HOME=/cache/torch \
    REPO_ID=/models/DeepSeek-OCR \
    OUTPUT_DIR=/outputs \
    PORT=8100

WORKDIR /app

# Make cache/output dirs (you’ll mount /models and /cache from host)
RUN mkdir -p /cache/hf /cache/torch /outputs

# ---- Python deps (DO NOT reinstall torch/torchvision; they come from the image) ----
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      "transformers==4.46.3" \
      "tokenizers==0.20.3" \
      einops \
      addict \
      easydict \
      python-multipart \
      safetensors \
      sentencepiece \
      fastapi \
      uvicorn \
      pillow \
      peft

# (Optional) If you *later* want FlashAttention, you can try this.
# For now, leave it out to avoid build headaches on new GPUs.
# RUN pip install "flash-attn==2.7.3" --no-build-isolation || true

# Your FastAPI app
COPY app.py /app/

EXPOSE 8100

# Uvicorn entrypoint
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
