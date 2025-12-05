
import os
import io
import tempfile
import numpy as np
from typing import List
from PIL import Image, ImageFilter, ImageOps
import torch
# ============================================================
# CONFIG (Optimized for SPEED + Realism)
# ============================================================
BASE_SAVE_DIR = "/content/drive/MyDrive/Article_Image_Generator"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
SUM_MODEL = "google/flan-t5-base"
N_CANDIDATES = 1          # ⚡ FASTEST: only 1 image per prompt
IMG_HEIGHT = 640          # ⚡ lower res = faster
IMG_WIDTH = 640
SAMPLER_STEPS = 18        # ⚡ reduced steps (still good quality)
CFG_SCALE = 7.0           # balanced
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STYLE = (
    "photorealistic, ultra-detailed, cinematic lighting, natural skin tones, "
    "film grain, 85mm lens, depth of field, award-winning photography"
)
NEGATIVE = (
    "lowres, blurry, deformed face, extra limbs, watermark, text, artifacts, bad anatomy"
)
# ============================================================
# TEXT EXTRACTION
# ============================================================
try: import pdfplumber
except: pdfplumber = None
try: import docx
except: docx = None
def extract_text_from_uploaded(file_obj, filename):
    ext = filename.split(".")[-1].lower()
    file_obj.seek(0)
    if ext == "txt":
        data = file_obj.read()
        return data.decode(errors="ignore") if isinstance(data, bytes) else str(data)
    if ext == "pdf" and pdfplumber:
        try:
            with pdfplumber.open(file_obj) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except:
            return ""
    if ext == "docx" and docx:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(file_obj.read()); tmp.close()
        d = docx.Document(tmp.name)
        os.unlink(tmp.name)
        return "\n".join([p.text for p in d.paragraphs])
    return ""
# ============================================================
# SUMMARIZATION
# ============================================================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
_tok = None
_summ = None
def _ensure_summarizer():
    global _tok, _summ
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(SUM_MODEL)
    if _summ is None:
        _summ = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL).to(DEVICE)
def summarize_article(text, max_scenes=6):
    _ensure_summarizer()
    prompt = f"Summarize into {max_scenes} cinematic scenes:\n{text}"
    inp = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    out = _summ.generate(inp.input_ids, max_length=256, num_beams=4)
    summary = _tok.decode(out[0], skip_special_tokens=True)
    lines = [l.strip() for l in summary.split("\n") if l.strip()]
    return "\n".join(lines[:max_scenes])
def extract_characters(text):
    _ensure_summarizer()
    prompt = "List main characters with 1–2 word physical descriptors:\n" + text
    inp = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    out = _summ.generate(inp.input_ids, max_length=80, num_beams=3)
    chars = _tok.decode(out[0], skip_special_tokens=True)
    return chars.replace("\n", " ").strip()
# ============================================================
# PROMPT BUILDER
# ============================================================
def build_prompts_from_summary(summary, characters, max_frames=6):
    lines = [l.strip() for l in summary.split("\n") if l.strip()][:max_frames]
    return [f"{l}. Characters: {characters}. {STYLE}" for l in lines]
# ============================================================
# MODEL LOADING (Optimized for speed)
# ============================================================
from diffusers import DiffusionPipeline
_pipe = None
def _load_model_fast():
    global _pipe
    models = [
        "Lykon/dreamshaper-xl-1.0",     # ⚡ fastest realistic
        "digiplay/AbsoluteReality_v1.8.1",
    ]
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    for m in models:
        try:
            print("Loading →", m)
            _pipe = DiffusionPipeline.from_pretrained(m, torch_dtype=dtype)
            _pipe.to(DEVICE)
            print("Loaded:", m)
            return
        except Exception as e:
            print("FAILED:", m, "→", e)
    raise RuntimeError("No model could be loaded.")
def get_pipe():
    global _pipe
    if _pipe is None:
        _load_model_fast()
    return _pipe
# ============================================================
# SIMPLE SHARPENING
# ============================================================
def enhance(img):
    try:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=110))
    except:
        pass
    return img
# ============================================================
# IMAGE GENERATION (FAST)
# ============================================================
def generate_high_realism_images_from_prompts(prompts, article_name="article"):
    pipe = get_pipe()
    save_dir = os.path.join(BASE_SAVE_DIR, article_name.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    out_paths = []
    for i, p in enumerate(prompts):
        gen = torch.Generator(DEVICE).manual_seed(torch.randint(0, 999999999, (1,)).item())
        img = pipe(
            prompt=p,
            negative_prompt=NEGATIVE,
            height=IMG_HEIGHT,
            width=IMG_WIDTH,
            guidance_scale=CFG_SCALE,
            num_inference_steps=SAMPLER_STEPS,
            generator=gen
        ).images[0]
        img = enhance(img)
        out = os.path.join(save_dir, f"scene_{i+1}.png")
        img.save(out)
        out_paths.append(out)
    return out_paths
def generate_images_for_prompts(prompts, article_name):
    return generate_high_realism_images_from_prompts(prompts, article_name)
