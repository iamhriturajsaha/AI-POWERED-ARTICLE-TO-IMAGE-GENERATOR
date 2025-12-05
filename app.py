import streamlit as st
import os
# -------------------------------
# Custom Glassmorphism CSS
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff !important;
}
.main, .block-container {
    background: transparent !important;
}
.block-container {
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.08) !important;
    padding: 35px 35px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.18);
    margin-top: 30px;
}
h1, h2, h3, h4, h5 {
    color: #e8f1f2 !important;
    font-weight: 700;
    text-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
.stRadio > label {
    color: white !important;
}
.stButton>button {
    background: linear-gradient(90deg, #ff7eb3, #ff758c);
    color: white !important;
    border-radius: 12px;
    padding: 0.65rem 1.3rem;
    font-size: 1rem;
    border: none;
    font-weight: 600;
    transition: 0.25s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff9eb9, #ff8aa7);
    transform: scale(1.05);
}
.stFileUploader label {
    color: #ffffff !important;
    font-size: 1.05rem !important;
}
hr {
    border: 1px solid rgba(255,255,255,0.25);
}
</style>
""", unsafe_allow_html=True)
# -------------------------------
# Title + Description
# -------------------------------
st.title("🌌 AI Article → Realistic Image Generator")
st.markdown("#### Convert entire articles into stunning cinematic images using AI 🎞️✨")
st.markdown("---")
# -------------------------------
# Import backend functions
# -------------------------------
from main import (
    extract_text_from_uploaded, 
    summarize_article,
    extract_characters,
    build_prompts_from_summary,
    generate_images_for_prompts
)
# -------------------------------
# UI Mode Selection
# -------------------------------
mode = st.radio("Select Mode", ["🎯 Single Article", "📚 Batch Processing"])
# ====================================================================
# SINGLE ARTICLE MODE
# ====================================================================
if mode == "🎯 Single Article":
    file = st.file_uploader("📄 Upload an Article File", type=["txt","pdf","docx"])
    if file:
        with st.spinner("📘 Reading file..."):
            text = extract_text_from_uploaded(file, file.name)
        st.subheader("📝 Article Preview")
        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))
        st.markdown("---")
        if st.button("🚀 Generate Cinematic Images"):
            with st.spinner("🧠 Understanding & summarizing article..."):
                summary = summarize_article(text)
                characters = extract_characters(text)
                prompts = build_prompts_from_summary(summary, characters)
            st.subheader("🎬 Generated Prompts")
            for i, p in enumerate(prompts):
                st.markdown(f"**Scene {i+1}:** {p}")
            st.markdown("---")
            with st.spinner("🎨 Creating high-quality realistic images..."):
                images = generate_images_for_prompts(prompts, article_name=file.name.split('.')[0])
            st.success("✨ Image Generation Complete!")
            for img in images:
                st.image(img, width=450)
# ====================================================================
# BATCH MODE
# ====================================================================
else:
    files = st.file_uploader("📄 Upload Multiple Article Files", type=["txt","pdf","docx"], accept_multiple_files=True)
    if files and st.button("🚀 Process All Files"):
        for file in files:
            st.write(f"📂 Processing **{file.name}**...")
            text = extract_text_from_uploaded(file, file.name)
            summary = summarize_article(text)
            characters = extract_characters(text)
            prompts = build_prompts_from_summary(summary, characters)
            images = generate_images_for_prompts(prompts, article_name=file.name.split('.')[0])
            st.success(f"✔ Finished {file.name}, generated {len(images)} images.")
        st.success("🎉 Batch Processing Complete! Images saved to Google Drive.")
