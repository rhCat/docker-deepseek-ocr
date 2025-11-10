# app.py
import os
import io
import uuid
import json
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
    output_dir: Optional[str] = None         # per-request folder

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

def _extract_text_from_infer(raw_ret: Any) -> str:
    """
    Try to get the main OCR text out of whatever infer() returned.
    Be tolerant of dict / list / str / bytes.
    """
    # Common: dict with "text" or "texts"
    if isinstance(raw_ret, dict):
        if isinstance(raw_ret.get("text"), str):
            return raw_ret["text"]
        if isinstance(raw_ret.get("texts"), list):
            return "\n\n".join(str(t) for t in raw_ret["texts"])
        # last resort: dump the whole thing
        return json.dumps(raw_ret, ensure_ascii=False, indent=2)

    # Sometimes a list of things
    if isinstance(raw_ret, (list, tuple)):
        parts = []
        for item in raw_ret:
            parts.append(_extract_text_from_infer(item))
        return "\n\n".join(p for p in parts if p.strip())

    # Raw bytes
    if isinstance(raw_ret, bytes):
        return raw_ret.decode("utf-8", errors="ignore")

    # Plain string (likely your case)
    if isinstance(raw_ret, str):
        return raw_ret.strip()

    # Fallback
    return str(raw_ret)


def _call_infer(
    img_path: str,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    test_compress: bool,
    save_results: bool,
    output_dir: Optional[pathlib.Path] = None,
) -> OCRResponse:
    model, tokenizer = load_model_and_tokenizer()

    # Make per-call directory
    if output_dir is None:
        call_id = uuid.uuid4().hex
        call_dir = OUTPUT_DIR / call_id
    else:
        call_dir = output_dir

    call_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_ret = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path=str(call_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=test_compress,
            save_results=save_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"infer() failed: {e}")

    # Collect all files for this call (even if there are none)
    saved: List[str] = []
    try:
        for p in call_dir.rglob("*"):
            if p.is_file():
                saved.append(str(p))
    except Exception:
        pass

    # ---- try to load the main OCR text from saved files ----
    main_text = ""
    try:
        candidates = []
        for p in call_dir.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            # include .mmd here
            if suf in (".md", ".mmd", ".txt", ".json"):
                candidates.append(p)

        for p in sorted(candidates, key=lambda x: x.name):
            try:
                main_text = p.read_text(encoding="utf-8", errors="ignore")
                if main_text.strip():
                    break
            except Exception:
                continue
    except Exception:
        pass

    # Fallback if still empty
    if not main_text:
        if isinstance(raw_ret, bytes):
            main_text = raw_ret.decode("utf-8", errors="ignore")
        elif isinstance(raw_ret, str):
            main_text = raw_ret
        elif raw_ret is None:
            main_text = ""
        else:
            main_text = ""
    return OCRResponse(
        text=main_text,
        saved_paths=saved,
        prompt=prompt,
        output_dir=str(call_dir),
    )

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
            # output_dir=None -> auto /outputs/<uuid>
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
    # One top-level dir for this batch call
    batch_id = uuid.uuid4().hex
    batch_dir = OUTPUT_DIR / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    results: List[OCRResponse] = []
    tmp_files: List[str] = []

    try:
        for idx, f in enumerate(files, start=1):
            img_path = _save_upload_to_tmp(f)
            tmp_files.append(img_path)

            # Subfolder per image
            img_dir = batch_dir / f"item_{idx:04d}"
            img_dir.mkdir(parents=True, exist_ok=True)

            res = _call_infer(
                img_path=img_path,
                prompt=prompt,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                test_compress=test_compress,
                save_results=save_results,
                output_dir=img_dir,
            )
            results.append(res)
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass

    return OCRBatchResponse(results=results)
