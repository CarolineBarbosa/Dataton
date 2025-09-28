import os
import sys
import yaml
import streamlit as st
import pandas as pd
import subprocess
# Ensure the 'ollama' package is installed
try:
    from ollama import Ollama
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
# Make workspace root importable so `from src.*` works when running with Streamlit
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.embedding_manager import EmbeddingManager
from src.indexer import FAISSIndexer
from src.recruiter import RecruiterBot, find_top_applicants_with_filters
from fpdf import FPDF


# Config paths
INDEX_CFG_PATH = os.path.join("src", "config", "index_config.yaml")
MODELS_CFG_PATH = os.path.join("models_config.yaml")
APPLICANTS_PATH = os.path.join("data", "processed", "applicants.parquet")

st.set_page_config(page_title="Recruiter Chat", layout="wide")

st.title("Recruiter Chat — Find Top Applicants")

# Load index config and applicants data
if not os.path.exists(INDEX_CFG_PATH):
    st.error(f"Index config not found: {INDEX_CFG_PATH}")
    st.stop()
with open(INDEX_CFG_PATH, "r") as f:
    index_cfg = yaml.safe_load(f)

if not os.path.exists(APPLICANTS_PATH):
    st.error(f"Applicants file not found at {APPLICANTS_PATH}")
    st.stop()
applicants_df = pd.read_parquet(APPLICANTS_PATH)

# Initialize manager/indexer once per session
if "emb_mgr" not in st.session_state:
    st.session_state.emb_mgr = EmbeddingManager(config_path=MODELS_CFG_PATH)
if "indexer" not in st.session_state:
    st.session_state.indexer = FAISSIndexer(index_cfg)
if "recruiter_bot" not in st.session_state:
    st.session_state.recruiter_bot = RecruiterBot(st.session_state.emb_mgr, st.session_state.indexer)

emb_mgr = st.session_state.emb_mgr
indexer = st.session_state.indexer
bot = st.session_state.recruiter_bot

# Chat window
st.subheader("Chat with Recruiter Bot")
chat_input = st.text_area("Enter job description or message", height=150)
if st.button("Find Candidates"):
    if not chat_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Processing..."):
            reply = bot.chat(chat_input)

            # Show top matches
            results = find_top_applicants_with_filters(
                job_description=bot.job_description,
                faiss_indexer=indexer,
                emb_mgr=emb_mgr,
                filters=bot.filters if bot.filters else None,
                top_n=5,
                search_k=200
            )
        if results:
            st.markdown("**Top matches:**")
            df_out = []
            for r in results:
                # Generate a PDF for the candidate's CV
                applicant_idx = r["applicant_idx"]
                applicant_info = applicants_df.iloc[applicant_idx] if applicant_idx < len(applicants_df) else {}

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

                pdf.cell(0, 10, f"Name: {r.get('nome', applicant_info.get('nome', 'N/A'))}", ln=True)
                pdf.cell(0, 10, f"Email: {applicant_info.get('email', 'N/A')}", ln=True)
                pdf.cell(0, 10, f"Phone: {applicant_info.get('phone', 'N/A')}", ln=True)
                pdf.multi_cell(0, 10, f"Additional Info: {r.get('metadata', {}).get('additional_info', 'N/A')}")
                pdf.cell(0, 10, f"Score: {r.get('score')}", ln=True)
                # Add metadata as sections in the PDF
                metadata = r.get('metadata', {})
                for key, value in metadata.items():
                    if key in ['titulo_vaga','nivel_profissional', 'nivel_academico','text',  'areas_atuacao', 'principais_atividades','competencia_tecnicas_e_comportamentais', 'nivel_ingles', 'nivel_espanhol', 'cidade' ]:
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, f"{key.capitalize()}:", ln=True)
                        pdf.set_font('Arial', '', 12)
                        # Replace unsupported characters with a safe alternative
                        safe_value = str(value).encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 10, safe_value)
                # Save the PDF
                pdf_output_path = os.path.join("output", f"{r.get('nome', 'candidate')}_cv.pdf")
                os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
                pdf.output(pdf_output_path)

                print(f"Generated CV PDF: {pdf_output_path}")
                applicant_idx = r["applicant_idx"]
                applicant_info = applicants_df.iloc[applicant_idx] if applicant_idx < len(applicants_df) else {}
                # Append candidate details to the output table
                df_out.append({
                    "Name": r.get("nome", applicant_info.get("nome", "N/A")),
                    "Email": applicant_info.get("email", "N/A"),
                    "Phone": applicant_info.get("phone", "N/A"),
                    "Additional Info": r.get("metadata", {}).get("additional_info", "N/A"),
                    "Score": r.get("score"),
                    "CV PDF": f"[Download PDF]({pdf_output_path})"
                })
            for candidate in df_out:
                with st.container():
                    
                    col1, col2 = st.columns([3, 1])  # Adjust column widths as needed
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
                    with open(candidate['CV PDF'].split('(')[1].split(')')[0], "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    st.download_button(
                        label="Download cancidate CV",
                        data=pdf_data,
                        file_name=os.path.basename(candidate['CV PDF'].split('(')[1].split(')')[0]),
                        mime="application/pdf",
                        key=f"download_{candidate['Name']}_{candidate['Email']}",  # Ensure unique key by including email
                        icon="📥",
                
                        disabled=st.session_state.get("download_disabled", False)
                    )
                    st.markdown("---")
        else:
            st.info("No matches found.")
