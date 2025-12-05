# 🚀AI-Powered Article to Image Generator

## Overview

The AI-Powered Article to Image Generator is an end-to-end automated pipeline that transforms written content into professional-grade visual assets. Built for content creators, digital marketers and developers, this system leverages state-of-the-art AI models to generate custom images that perfectly complement article text—eliminating the need for expensive stock photography or manual design work.

### Key Capabilities

- **Intelligent Content Analysis** - Automatically extracts visual concepts from article text using FLAN-T5 summarization.
- **Enterprise-Grade Generation** - Powered by Stable Diffusion XL with multi-model fallback for reliability.
- **Zero-Setup Deployment** - Fully operational web interface deployed directly from Google Colab via Ngrok.
- **Production-Ready Output** - Organized file management with metadata tracking and Google Drive integration.
- **Batch Processing** - Generate dozens of images in a single session with consistent quality.

<p align="center">
  <img src="Streamlit Screenshots/1.png" alt="1" width="1000"/><br>
</p>
<table>
  <tr>
    <td align="center"><img src="Streamlit Screenshots/2.png" width="500"></td>
    <td align="center"><img src="Streamlit Screenshots/3.png" width="500"></td>
  </tr>
  <tr>
    <td align="center"><img src="Streamlit Screenshots/4.png" width="500"></td>
    <td align="center"><img src="Streamlit Screenshots/5.png" width="500"></td>
  </tr>
  <tr>
    <td align="center"><img src="Streamlit Screenshots/6.png" width="500"></td>
    <td align="center"><img src="Streamlit Screenshots/7.png" width="500"></td>
  </tr>
</table>

## Features

### 🎨 Advanced Image Generation

**Multi-Model Support with Intelligent Fallback**
- Primary - Stable Diffusion XL (highest quality).
- Secondary - SDXL-Turbo (4× faster, optimized inference).
- Tertiary - SD 1.5 (maximum compatibility).
- Automatic switching on VRAM limitations or model failures.

**Hardware Optimization**
- Automatic GPU detection and configuration.
- FP16 precision for GPU, FP32 for CPU.
- Memory-efficient settings (attention slicing, VAE tiling, CPU offload).
- Configurable parameters - resolution, inference steps, guidance scale, samplers.

### 🧠 Smart Prompt Engineering

**Automated Text-to-Prompt Conversion**
- FLAN-T5 base model (250M parameters) for semantic summarization.
- Extracts visual concepts, mood and key elements from prose.
- Enhances prompts with quality modifiers and style descriptors.
- Maintains thematic consistency across multi-section articles.

**Example Transformation**
```
Input - "Renewable energy adoption has transformed urban infrastructure..."
Output - "Modern city with solar panels, sustainable architecture, urban skyline, professional photography, high detail"
```

### 📦 Batch Processing & Organization

**Scalable Generation Pipeline**
- Process entire articles with multiple sections.
- Generate 1 to 100+ images per session.
- Automatic folder structure creation.
- Timestamp-based file naming with collision handling.
- Google Drive integration for persistent storage.

### 🌐 Professional Web Interface

**Full-Featured Streamlit Application**
- Clean, intuitive UI with real-time feedback.
- Article text input with file upload support (TXT, PDF, DOCX).
- Interactive configuration sidebar (resolution, steps, batch size).
- Live generation progress tracking.
- Image gallery with preview and download.
- Bulk download as ZIP.

**Secure Public Access**
- Ngrok tunnel provides HTTPS endpoint.
- No local port forwarding required.
- Optional password protection.
- Session persistence.

### 🛡️ Safety & Reliability

**Content Filtering**
- Default negative prompts exclude NSFW, violence and disturbing content.
- Customizable safety parameters.
- Filename sanitization prevents injection attacks.

**Error Handling**
- Graceful degradation on model failures.
- Automatic retry logic.
- Detailed error logging.
- User-friendly error messages.

## Architecture

The system follows a modular, layered architecture designed for maintainability and extensibility -
```
┌─────────────────────────────────────────────────────┐
│              User Interface (Streamlit)             │
│                    via Ngrok Tunnel                 │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────┐
│            Application Logic Layer                  │
│  • Input Validation  • Session Management           │
│  • Request Routing   • File Handling                │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────┐
│           Processing Pipeline Layer                 │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐   │
│  │Text Parser  │→ │ FLAN-T5     │→ │Prompt      │   │
│  │(Sections)   │  │ Summarizer  │  │Enhancement │   │
│  └─────────────┘  └─────────────┘  └────────────┘   │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────┐
│            Model Inference Layer                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Model Manager (Fallback Logic)              │   │
│  │  ├─ SDXL (Primary)                           │   │
│  │  ├─ SDXL-Turbo (Secondary)                   │   │
│  │  └─ SD 1.5 (Tertiary)                        │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Hardware Abstraction                        │   │
│  │  • GPU/CPU Detection  • Memory Management    │   │
│  └──────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────┐
│          Storage & Output Management                │
│  • Google Drive Integration  • Metadata Tracking    │
│  • File Naming & Organization  • Collision Handling │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Google account (for Colab and Drive).
- Ngrok account (https://ngrok.com) (free tier).
- Stable internet connection.

### Launch in 3 Steps

1. **Open the Colab Notebook**
   - Access the `.ipynb` file in this repository.
   - Click "Open in Colab".
   - Select GPU runtime - **Runtime → Change runtime type → T4 GPU**.

2. **Configure Ngrok**
   ```python
   NGROK_AUTH_TOKEN = "your_token_here"
   ```

3. **Run All Cells**
   - The notebook handles all setup automatically.
   - Access your app at the generated Ngrok URL.

## Installation

### Automatic Installation (Recommended)

The notebook automatically installs all dependencies -
```python
!pip install --quiet transformers diffusers accelerate safetensors sentencepiece streamlit pyngrok==7.0.0 pdfplumber python-docx bitsandbytes Pillow > /dev/null
```

### Manual Installation

For local deployment -
```bash
pip install transformers diffusers accelerate safetensors
pip install streamlit pdfplumber python-docx bitsandbytes Pillow pyngrok
```

### Model Downloads

Models are automatically downloaded on first run (~20 GB total) -

| Model | Size | Purpose |
|-------|------|---------|
| `stabilityai/stable-diffusion-xl-base-1.0` | 6.94 GB | Primary image generation |
| `stabilityai/sdxl-turbo` | 6.94 GB | Fast inference |
| `runwayml/stable-diffusion-v1-5` | 4.27 GB | Fallback model |
| `google/flan-t5-base` | 892 MB | Text summarization |

**Storage Requirements** - 25 GB free space recommended.

## Usage

### Basic Workflow

1. **Launch the Application**
   - Execute all notebook cells.
   - Copy the Ngrok URL from output.

2. **Access Web Interface**
   - Open the Ngrok URL in your browser.
   - You'll see the Streamlit dashboard.

3. **Input Article Content**
   - Paste text directly into the text area.
   - Or upload a file (TXT, PDF, DOCX).

4. **Configure Settings**
   - Adjust resolution in sidebar (default - 1024×1024).
   - Set inference steps (default - 30).
   - Modify batch size (default - 1 image per prompt).

5. **Generate Images**
   - Click "Generate Images".
   - Monitor progress in real-time.
   - View results in the gallery.

6. **Download Results**
   - Download individual images.
   - Or use "Download All as ZIP".

### Advanced Usage

#### Programmatic Generation
```python
from main import generate_images_for_prompts
# Custom prompts
prompts = [
    "A cyberpunk cityscape at night, neon lights, cinematic",
    "Ancient library interior, mystical atmosphere, warm lighting"
]
# Generate images
generate_images_for_prompts(
    prompts=prompts,
    article_name="SciFi_Article",
    num_images_per_prompt=2,
    steps=40,
    resolution=(1024, 1024)
)
```

#### Batch Processing from File
```python
# Load article
with open('article.txt', 'r') as f:
    article_text = f.read()
# Split into sections
sections = article_text.split('\n\n')
# Generate prompts automatically
from main import generate_prompt_from_text
prompts = [generate_prompt_from_text(section) for section in sections]
# Generate all images
generate_images_for_prompts(prompts, article_name="Article_Name")
```

#### Style Presets
```python
STYLE_PRESETS = {
    "photorealistic": "professional photography, 8k, highly detailed",
    "artistic": "digital art, concept art, trending on artstation",
    "cinematic": "cinematic lighting, dramatic, film still",
    "minimalist": "minimalist design, clean, simple, modern"
}
# Apply style
generate_images_for_prompts(
    prompts=["A mountain sunset"],
    style_preset="cinematic"
)
```

## Configuration

### Model Selection
```python
MODEL_CONFIG = {
    "primary_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "secondary_model": "stabilityai/sdxl-turbo",
    "fallback_model": "runwayml/stable-diffusion-v1-5"
}
```

### Generation Parameters
```python
GENERATION_CONFIG = {
    "default_resolution": (1024, 1024),
    "default_steps": 30,
    "default_guidance_scale": 7.5,
    "negative_prompt": "blurry, low quality, distorted, nsfw, violent, gore"
}
```

### Optimization Settings
```python
OPTIMIZATION_CONFIG = {
    "enable_attention_slicing": True,  # Reduces VRAM usage
    "enable_vae_slicing": True,        # Reduces VRAM usage
    "enable_cpu_offload": False,       # Enable if VRAM < 8GB
    "use_tf32": True                   # Faster on Ampere GPUs
}
```

### Output Configuration
```python
OUTPUT_CONFIG = {
    "base_dir": "/content/drive/MyDrive/Article_Image_Generator",
    "save_format": "PNG",
    "embed_metadata": True,
    "create_thumbnails": False
}
```

## Technical Details

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Deep Learning Framework** | PyTorch | 2.0+ |
| **Diffusion Models** | Hugging Face Diffusers | 0.21+ |
| **Text Models** | Transformers | 4.30+ |
| **Web Framework** | Streamlit | 1.28+ |
| **Tunneling** | Pyngrok | 6.0+ |
| **Image Processing** | Pillow | 10.0+ |
| **Acceleration** | Accelerate | 0.20+ |

### Model Specifications

#### Stable Diffusion XL
- **Architecture** - Latent Diffusion with dual text encoders.
- **Parameters** - 3.5B.
- **Optimal Resolution** - 1024×1024.
- **Training Data** - LAION-5B (filtered).
- **Performance** - 8 to 12s per image @ 30 steps (T4 GPU).

#### SDXL-Turbo
- **Architecture** - Distilled SDXL.
- **Parameters** - 3.5B.
- **Optimal Steps** - 4 (vs. 30 for SDXL).
- **Performance** - 2 to 3s per image @ 4 steps (T4 GPU).

#### FLAN-T5 Base
- **Architecture** - Encoder-Decoder Transformer.
- **Parameters** - 250M.
- **Training** - Instruction fine-tuned on 1,000+ tasks.
- **Performance** - <100ms per summarization.

### Hardware Requirements

**Minimum**
- CPU - Any modern processor.
- RAM - 8 GB.
- Storage - 25 GB free.
- Internet - Stable broadband.

**Recommended**
- GPU - NVIDIA T4 or better (16 GB VRAM).
- RAM - 16 GB.
- Storage - 50 GB SSD.
- Internet - 10+ Mbps.

**Performance Benchmarks** (T4 GPU)
- SDXL @ 1024×1024, 30 steps - 8 to 12s per image.
- SDXL-Turbo @ 1024×1024, 4 steps - 2 to 3s per image.
- SD 1.5 @ 512×512, 50 steps - 5 to 7s per image.

## Troubleshooting

#### a) CUDA Out of Memory

**Symptoms** - `RuntimeError - CUDA out of memory`.

**Solutions**
1. Reduce resolution to 768×768 or 512×512.
2. Decrease inference steps to 20.
3. Enable CPU offloading -
   ```python
   pipe.enable_model_cpu_offload()
   ```
4. Switch to SDXL-Turbo or SD 1.5.
5. Restart notebook to clear VRAM.

#### b) Ngrok Tunnel Not Starting

**Symptoms** - No public URL generated.

**Solutions**
1. Verify your Ngrok auth token is correct.
2. Check for existing tunnels -
   ```python
   !pkill ngrok
   ```
3. Ensure no firewall blocking port 8501.
4. Try restarting the Colab runtime.

#### c) Low-Quality or Abstract Images

**Symptoms** - Images don't match prompts or look distorted.

**Solutions**
1. Increase inference steps to 40-50.
2. Raise guidance scale to 9-12.
3. Improve prompt detail and specificity.
4. Add negative prompts - "blurry, distorted, low quality".
5. Ensure you're using SDXL (not Turbo) for quality.

#### d) Slow Generation Speed

**Symptoms** - Each image takes >30 seconds.

**Solutions**
1. Verify GPU is enabled in Colab (Runtime → Change runtime type).
2. Check GPU utilization - `!nvidia-smi`.
3. Switch to SDXL-Turbo for faster inference.
4. Reduce resolution.
5. Consider using a higher-tier Colab subscription (A100 GPU).

#### e) Model Download Failures

**Symptoms** - `OSError - Can't load model`.

**Solutions**
1. Check internet connection.
2. Clear Hugging Face cache -
   ```python
   !rm -rf /root/.cache/huggingface
   ```
3. Manually specify model revision - 
   ```python
   pipe = DiffusionPipeline.from_pretrained(
       model_id,
       revision="main",
       torch_dtype=torch.float16
   )
   ```

## Future Scope

 - Multi-Model Support & Dynamic Model Switching.
 - Prompt Engineering Enhancements.
 - Full Article-to-Storyboard Automation.
 - Vision-Language Feedback Loop.
 - Integration With CMS Platforms.

