import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume vs JD Analyzer", layout="centered")

st.title("📄 Resume vs Job Description Analyzer")
st.markdown("Upload your resume and paste a job description. Get your match score instantly!")

# --- Helper: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- Helper: Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# --- Upload Resume
resume_file = st.file_uploader("📤 Upload your Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description here")

if st.button("Analyze Match"):
    if resume_file is not None and job_description.strip() != "":
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description)

            # TF-IDF + Cosine Similarity
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_clean, jd_clean])
            score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            st.success(f"✅ Match Score: **{score * 100:.2f}%**")

            # Show missing keywords (basic method)
            resume_words = set(resume_clean.split())
            jd_words = set(jd_clean.split())
            missing_keywords = jd_words - resume_words

            if missing_keywords:
                st.markdown("🛠️ **Consider adding these keywords from the JD:**")
                st.write(", ".join(list(missing_keywords)[:20]))
            else:
                st.markdown("🎯 Your resume covers most keywords in the JD!")
    else:
        st.warning("Please upload a resume and paste the job description.")
