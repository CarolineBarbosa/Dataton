import os
import sys
import yaml
import streamlit as st
import pandas as pd
import streamlit_ext as ste
from fpdf import FPDF

# Ensure the 'ollama' package is installed
# def ensure_package_installed(package_name):
#     try:
#         __import__(package_name)
#     except ImportError:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# ensure_package_installed("ollama")

# Add workspace root to sys.path
def add_workspace_to_path():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

add_workspace_to_path()

from src.embedding_manager import EmbeddingManager
from src.indexer import FAISSIndexer
from src.recruiter import RecruiterBot, find_top_applicants_with_filters

# Config paths
INDEX_CFG_PATH = os.path.join("src", "config", "index_config.yaml")
MODELS_CFG_PATH = os.path.join("models_config.yaml")
APPLICANTS_PATH = os.path.join("data", "processed", "applicants.parquet")

# Streamlit setup
def setup_streamlit():
    st.set_page_config(page_title="Recruiter Chat", layout="wide")
    st.title("Recruiter Chat — Find Top Applicants")

setup_streamlit()

# Load configuration files
def load_config(path, error_message):
    if not os.path.exists(path):
        st.error(error_message)
        st.stop()
    with open(path, "r") as f:
        return yaml.safe_load(f)

index_cfg = load_config(INDEX_CFG_PATH, f"Index config not found: {INDEX_CFG_PATH}")
applicants_df = pd.read_parquet(APPLICANTS_PATH) if os.path.exists(APPLICANTS_PATH) else st.error(f"Applicants file not found at {APPLICANTS_PATH}")

# Initialize session state
def initialize_session_state():
    if "emb_mgr" not in st.session_state:
        st.session_state.emb_mgr = EmbeddingManager(config_path=MODELS_CFG_PATH)
    if "indexer" not in st.session_state:
        st.session_state.indexer = FAISSIndexer(index_cfg)
    if "recruiter_bot" not in st.session_state:
        st.session_state.recruiter_bot = RecruiterBot(st.session_state.emb_mgr, st.session_state.indexer)

initialize_session_state()

emb_mgr = st.session_state.emb_mgr
indexer = st.session_state.indexer
bot = st.session_state.recruiter_bot

# Generate PDF for a candidate
def generate_candidate_pdf(candidate, applicant_info):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Candidate CV', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    pdf.cell(0, 10, f"Name: {candidate.get('nome', applicant_info.get('nome', 'N/A'))}", ln=True)
    pdf.cell(0, 10, f"Email: {applicant_info.get('email', 'N/A')}", ln=True)
    pdf.cell(0, 10, f"Phone: {applicant_info.get('phone', 'N/A')}", ln=True)
    pdf.multi_cell(0, 10, f"Additional Info: {candidate.get('metadata', {}).get('additional_info', 'N/A')}")
    pdf.cell(0, 10, f"Score: {candidate.get('score')}", ln=True)

    metadata = candidate.get('metadata', {})
    for key, value in metadata.items():
        if key in ['titulo_vaga', 'nivel_profissional', 'nivel_academico', 'text', 'areas_atuacao', 'principais_atividades', 'competencia_tecnicas_e_comportamentais', 'nivel_ingles', 'nivel_espanhol', 'cidade']:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"{key.capitalize()}:", ln=True)
            pdf.set_font('Arial', '', 12)
            safe_value = str(value).encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, safe_value)

    pdf_output_path = os.path.join("output", f"{candidate.get('nome', 'candidate')}_cv.pdf")
    os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
    pdf.output(pdf_output_path)
    return pdf_output_path

# Display candidate details
def display_candidate(candidate, applicant_info, pdf_path):
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Name:** {candidate['Name']}")
            st.markdown(f"**Email:** {candidate['Email']}")
        with col2:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                    <span style="font-weight: bold; color: #007BFF;">Score: {candidate['Score'] * 100:.0f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
        ste.download_button(
            label="Download candidate CV",
            data=pdf_data,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
        )
        st.markdown("---")

# Process chat input and find candidates
def process_chat_input(chat_input):
    if not chat_input.strip():
        st.warning("Please enter a message.")
        return

    with st.spinner("Processing..."):
        reply = bot.chat(chat_input)
        results = find_top_applicants_with_filters(
            job_description=bot.job_description,
            faiss_indexer=indexer,
            emb_mgr=emb_mgr,
            filters=bot.filters if bot.filters else None,
            top_n=10,
            search_k=200
        )

    if results:
        st.markdown("**Top matches:**")
        for r in results:
            applicant_idx = r["applicant_idx"]
            applicant_info = applicants_df.iloc[applicant_idx] if applicant_idx < len(applicants_df) else {}
            pdf_path = generate_candidate_pdf(r, applicant_info)
            display_candidate({
                "Name": r.get("nome", applicant_info.get("nome", "N/A")),
                "Email": applicant_info.get("email", "N/A"),
                "Phone": applicant_info.get("phone", "N/A"),
                "Additional Info": r.get("metadata", {}).get("additional_info", "N/A"),
                "Score": r.get("score"),
            }, applicant_info, pdf_path)
    else:
        st.info("No matches found.")

# Main chat interface
def main():
    st.subheader("Chat with Recruiter Bot")
    chat_input = st.text_area("Enter job description or message", height=150)
    if st.button("Find Candidates"):
        process_chat_input(chat_input)

main()
