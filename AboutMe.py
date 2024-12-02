from pathlib import Path
import streamlit as st
from PIL import Image


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "ANAND.2024.pdf"
profile_pic = current_dir / "assets" / "profile-pic.png"


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Portfolio | Anand Sahu"
PAGE_ICON = ":wave:"
NAME = "Anand Sahu"
DESCRIPTION = """
Aspiring Data Scientist passionate about leveraging Data Science, Machine Learning, AI, and Generative AI to solve real-world challenges and drive data-informed decisions.
"""

EMAIL = "anandsahu5097@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/anandsahu/",
    "GitHub": "https://github.com/anand23124",
    "Twitter": "https://x.com/sahu__anand_",
    "Medium": "https://medium.com/@anand_sahu"
}


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


# --- LOAD CSS, PDF & PROFILE PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📫", EMAIL)


# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")


# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("About Me")
st.write(
    """
- 🚀 Experienced in building AI-driven solutions like AI Chatbots and Image Extractors.
- 🛠️ Skilled in machine learning, IoT systems, and hardware-software integration.
- 🌟 Proficient in applying ML algorithms to diverse datasets.
- 🤝 Collaborative and eager to solve real-world problems with AI.
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- **Programming:** Python, SQL, OOP
- **Frameworks/Libraries:** PyTorch, Hugging Face, LangChain, Streamlit, Docker
- **AI/ML Expertise:** NLP, LLMs (OpenAI, Llama, Google Gemini, AWS Bedrock)
- **Databases:** FAISS, Chroma, ObjectBox, Astra
- **Tools:** VS Code, PyCharm, Jupyter, Git, GitHub
- **Cloud Platforms:** AWS (EC2, Lambda, Fargate, S3, Bedrock)
"""
)
