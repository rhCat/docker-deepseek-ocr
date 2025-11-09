# app.py
import os
import io
import uuid
import pathlib
import tempfile
from typing import List, Optional, Any

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel

from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# ------------------ Config ------------------
REPO_ID = os.getenv("REPO_ID", "deepseek-ai/DeepSeek-OCR")
# Set which GPU to use (e.g., "0" or "1"); empty = all visible
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Where to write any artifacts that DeepSeek-OCR saves
OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default prompt from authors’ example (you can override per request)
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "

# Model load knobs
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "1") == "1"

# ------------------ Lazy globals ------------------
_model: Optional[Any] = None
_tokenizer: Optional[AutoTokenizer] = None

def load_model_and_tokenizer():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    # Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)

    # Try FlashAttention2; fall back gracefully if not available
    model_kwargs = dict(
        trust_remote_code=True,
        use_safetensors=True,
    )
    if USE_FLASH_ATTENTION:
        model_kwargs["_attn_implementation"] = "flash_attention_2"

    try:
        _model = AutoModel.from_pretrained(REPO_ID, **model_kwargs)
    except TypeError:
        # Older transformers or missing flash-attn -> retry without FA2
        model_kwargs.pop("_attn_implementation", None)
        _model = AutoModel.from_pretrained(REPO_ID, **model_kwargs)

    _model = _model.eval()
    if DEVICE == "cuda":
        _model = _model.to(torch.bfloat16).cuda()

    # Small speed knobs (safe if CUDA present)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    return _model, _tokenizer

# ------------------ FastAPI ------------------
app = FastAPI(title="DeepSeek-OCR", version="1.0")

class OCRResponse(BaseModel):
    text: str
    saved_paths: Optional[List[str]] = None  # any files saved by the model
    prompt: Optional[str] = None

class OCRBatchResponse(BaseModel):
    results: List[OCRResponse]

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "repo": REPO_ID,
        "output_dir": str(OUTPUT_DIR),
    }

def _save_upload_to_tmp(upload: UploadFile) -> str:
    # Ensure it’s a real image
    raw = upload.file.read()
    upload.file.seek(0)
    try:
        Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    suffix = pathlib.Path(upload.filename or f"{uuid.uuid4()}.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name

def _call_infer(img_path: str, prompt: str, base_size: int, image_size: int,
                crop_mode: bool, test_compress: bool, save_results: bool) -> OCRResponse:
    model, tokenizer = load_model_and_tokenizer()

    # The authors’ API:
    # model.infer(tokenizer, prompt='', image_file='', output_path=' ',
    #             base_size=1024, image_size=640, crop_mode=True,
    #             test_compress=False, save_results=False)
    try:
        text = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path=str(OUTPUT_DIR),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=test_compress,
            save_results=save_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"infer() failed: {e}")

    # Collect any files newly written to OUTPUT_DIR (best-effort)
    saved = []
    try:
        # List files modified in the last 5 minutes as a heuristic
        import time
        cutoff = time.time() - 300
        for p in OUTPUT_DIR.glob("*"):
            try:
                if p.is_file() and p.stat().st_mtime >= cutoff:
                    saved.append(str(p))
            except Exception:
                pass
    except Exception:
        pass

    return OCRResponse(text=str(text), saved_paths=saved, prompt=prompt)

@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    image: UploadFile = File(...),
    prompt: str = Query(DEFAULT_PROMPT),
    base_size: int = Query(1024, ge=256, le=4096, description="See model README"),
    image_size: int = Query(640, ge=256, le=4096, description="See model README"),
    crop_mode: bool = Query(True),
    test_compress: bool = Query(True),
    save_results: bool = Query(True),
):
    img_path = _save_upload_to_tmp(image)
    try:
        return _call_infer(
            img_path=img_path,
            prompt=prompt,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=test_compress,
            save_results=save_results,
        )
    finally:
        try:
            os.unlink(img_path)
        except Exception:
            pass

@app.post("/ocr_batch", response_model=OCRBatchResponse)
async def ocr_batch(
    files: List[UploadFile] = File(...),
    prompt: str = Query(DEFAULT_PROMPT),
    base_size: int = Query(1024, ge=256, le=4096),
    image_size: int = Query(640, ge=256, le=4096),
    crop_mode: bool = Query(True),
    test_compress: bool = Query(True),
    save_results: bool = Query(True),
):
    results: List[OCRResponse] = []
    tmp_files: List[str] = []
    try:
        for f in files:
            img_path = _save_upload_to_tmp(f)
            tmp_files.append(img_path)
            res = _call_infer(
                img_path=img_path,
                prompt=prompt,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                test_compress=test_compress,
                save_results=save_results,
            )
            results.append(res)
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass
    return OCRBatchResponse(results=results)

