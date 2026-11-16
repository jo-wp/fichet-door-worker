"""Worker RunPod : édition d'image multi-références avec Qwen-Image-Edit-2511.

Entrée  : { prompt, images: [url | data URI | base64], negative_prompt?, width?, height?,
            steps? (40), cfg? (4.0), seed?, output_format? ("jpeg" | "png") }
          ou { action: "health" }
Sortie  : { image: <base64>, mime_type, width, height, seed, steps, cfg, seconds }
"""
import base64
import io
import os
import random
import time

# Cache Hugging Face : cache de modèles RunPod ou network volume s'il y en a un.
if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", "/runpod-volume/huggingface-cache")

import requests
import runpod
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image, ImageOps

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
MAX_SIDE = int(os.environ.get("MAX_SIDE", "2048"))
PIPE = None


def get_pipe():
    global PIPE
    if PIPE is None:
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 70:
            pipe.to("cuda")
        else:  # 48 Go et moins : les sous-modèles passent sur le GPU chacun leur tour
            pipe.enable_model_cpu_offload()
        PIPE = pipe
    return PIPE


def load_image(src):
    if src.startswith(("http://", "https://")):
        res = requests.get(src, timeout=60)
        res.raise_for_status()
        data = res.content
    else:
        data = base64.b64decode(src.split(",", 1)[1] if src.startswith("data:") else src)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))  # photos de téléphone pivotées
    if img.mode in ("RGBA", "LA", "P"):  # PNG transparent -> fond blanc
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def default_size(img):
    """Même ratio que la première image (la scène), grand côté 1536, multiples de 16."""
    scale = 1536 / max(img.size)
    return tuple(max(256, round(v * scale / 16) * 16) for v in img.size)


def handler(job):
    inp = job.get("input") or {}
    if inp.get("action") == "health":
        return {
            "status": "ok",
            "model": MODEL_ID,
            "loaded": PIPE is not None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hf_home": os.environ.get("HF_HOME"),
        }

    prompt = inp.get("prompt")
    sources = inp.get("images") or []
    if not prompt or not sources:
        return {"error": "'prompt' et 'images' (1 à 3 images) sont obligatoires"}
    if len(sources) > 3:
        return {"error": "3 images au maximum"}

    images = [load_image(s) for s in sources]
    width, height = default_size(images[0])
    width = int(inp.get("width") or width) // 16 * 16
    height = int(inp.get("height") or height) // 16 * 16
    steps = int(inp.get("steps") or 40)
    cfg = float(inp.get("cfg") or 4.0)
    seed = int(inp["seed"]) if inp.get("seed") is not None else random.randint(0, 2**31 - 1)
    fmt = "PNG" if str(inp.get("output_format", "jpeg")).lower() == "png" else "JPEG"

    start = time.time()
    with torch.inference_mode():
        result = get_pipe()(
            image=images,
            prompt=prompt,
            negative_prompt=inp.get("negative_prompt") or " ",
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, fmt, quality=95)
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "mime_type": "image/png" if fmt == "PNG" else "image/jpeg",
        "width": result.width,
        "height": result.height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "seconds": round(time.time() - start, 1),
    }


runpod.serverless.start({"handler": handler})
