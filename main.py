
import os
import pdfplumber
import docx
import torch
from diffusers import DiffusionPipeline
from pathlib import Path
from PIL import Image
# -------------------------------
# CONFIG
# -------------------------------
BASE_SAVE_DIR = "/content/drive/MyDrive/Article_Image_Generator"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
# -------------------------------
# TEXT EXTRACTION
# -------------------------------
def extract_text_from_uploaded(file_obj, filename):
    ext = filename.split(".")[-1].lower()
    if ext == "txt":
        data = file_obj.read()
        return data.decode(errors="ignore")
    elif ext == "pdf":
        with pdfplumber.open(file_obj) as pdf:
            texts = [p.extract_text() for p in pdf.pages if p.extract_text()]
        return "\n\n".join(texts)
    elif ext == "docx":
        tmp_path = "/content/tmp.docx"
        with open(tmp_path, "wb") as f:
            f.write(file_obj.read())
        doc = docx.Document(tmp_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""
# -------------------------------
# SUMMARIZATION
# -------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
SUM_MODEL = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(SUM_MODEL)
model_summ = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL).to("cuda")
def summarize_article(text):
    prompt = "Summarize this into 5 short scene descriptions:\n" + text
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    outputs = model_summ.generate(inputs.input_ids, max_length=250, num_beams=4)
    return tok.decode(outputs[0], skip_special_tokens=True)
def extract_characters(text):
    prompt = "List main characters with short physical descriptions:\n" + text
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    outputs = model_summ.generate(inputs.input_ids, max_length=120, num_beams=3)
    return tok.decode(outputs[0], skip_special_tokens=True)
# -------------------------------
# PROMPT BUILDER
# -------------------------------
def build_prompts_from_summary(summary, characters, max_frames=6):
    lines = [s.strip() for s in summary.split("\n") if len(s.strip()) > 10]
    lines = lines[:max_frames]
    prompts = []
    for s in lines:
        p = (
            f"Ultra-photorealistic cinematic image: {s}. "
            f"Characters: {characters}. "
            "8k detail, sharp focus, dramatic lighting, film still, award-winning photography."
        )
        prompts.append(p)
    return prompts
# -------------------------------
# IMAGE GENERATION
# -------------------------------
MODEL_ID = "digiplay/AbsoluteReality_v1.8.1"  # fast public model
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda")
def generate_images_for_prompts(prompts, article_name):
    save_dir = os.path.join(BASE_SAVE_DIR, article_name.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    out_paths = []
    for i, prompt in enumerate(prompts):
        img = pipe(prompt=prompt, height=768, width=768, num_inference_steps=25).images[0]
        path = os.path.join(save_dir, f"frame_{i+1}.png")
        img.save(path)
        out_paths.append(path)
    return out_paths
