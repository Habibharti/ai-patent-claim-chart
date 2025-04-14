import streamlit as st
from fpdf import FPDF
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
from io import BytesIO

# Load AI model with proper caching
@st.cache_resource
def load_generator():
    try:
        model_name = "tiiuae/falcon-rw-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

# Match claims with product features using basic overlap
def basic_match_claims(claim_elements, features):
    mappings = []
    for elem in claim_elements:
        elem_words = set(elem.lower().split())
        best_feature = max(features, key=lambda feat: len(elem_words & set(feat.lower().split())), default="No match found")
        mappings.append((elem, best_feature, None))
    return mappings

# AI-enhanced matcher
def ai_match_claims(claim_elements, features, generator):
    mappings = []
    for elem in claim_elements:
        prompt = f"Patent Claim Element:\n{elem}\n\nProduct Features:\n"
        for i, feat in enumerate(features, start=1):
            prompt += f"{i}. {feat}\n"
        prompt += "\nQuestion: Which product feature best satisfies this claim element and why?"

        try:
            output = generator(prompt, max_new_tokens=100, do_sample=False)[0]['generated_text']
        except Exception as e:
            st.warning(f"AI failed for: {elem}. Error: {e}")
            return basic_match_claims(claim_elements, features)

        m = re.match(r"(\d+)[\).:\s-]*", output.strip())
        if m and 0 < int(m.group(1)) <= len(features):
            feature = features[int(m.group(1)) - 1]
        else:
            feature = features[0]
        mappings.append((elem, feature, output.strip()))
    return mappings

# PDF class with header/footer
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, "AI-Driven Patent Claim Chart Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

# Generate clean and structured PDF
def create_claim_chart_pdf(mappings, mode):
    pdf = PDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Table of Contents", ln=True)
    pdf.ln(5)
    
    toc = []
    for i in range(len(mappings)):
        toc.append((pdf.page_no() + i + 1, f"Claim {i + 1}"))
    
    start_page = pdf.page_no() + 1
    for i, (elem, feat, justify) in enumerate(mappings, start=1):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{i}. Patent Claim Element:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(190, 8, elem)
        pdf.ln(2)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Matching Product Feature:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(190, 8, feat)
        pdf.ln(2)

        if mode == "AI-Enhanced" and justify:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Justification:", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(190, 8, justify)
        pdf.ln(4)

    pdf.page = 1
    pdf.set_y(25)
    pdf.set_font("Arial", '', 11)
    for page_num, label in toc:
        pdf.cell(0, 10, f"{label} .......................................... Page {page_num}", ln=True)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Streamlit UI
st.title("AI-Driven Patent Claim Chart Generator")

st.markdown("""
Upload or enter your patent claims and product features. Choose between basic or AI-enhanced mapping. Download results as a structured PDF.
""")

claims_input = st.text_area("Patent Claims (one per line or separated by semicolon)", height=150)
features_input = st.text_area("Product Features (one per line)", height=150)
mode = st.radio("Mapping Mode", ["Basic", "AI-Enhanced"], index=1)

if st.button("Generate Claim Chart"):
    claims = [x.strip() for x in claims_input.replace(';', '\n').splitlines() if x.strip()]
    features = [x.strip() for x in features_input.splitlines() if x.strip()]

    if not claims or not features:
        st.error("Please enter both claims and product features.")
    else:
        if mode == "AI-Enhanced":
            generator = load_generator()
            if generator:
                with st.spinner("Running AI model..."):
                    mappings = ai_match_claims(claims, features, generator)
            else:
                st.warning("AI unavailable. Using fallback.")
                mappings = basic_match_claims(claims, features)
                mode = "Basic"
        else:
            mappings = basic_match_claims(claims, features)

        st.subheader("Generated Claim Chart")
        for elem, feat, justify in mappings:
            st.markdown(f"**Claim Element:** {elem}")
            st.markdown(f"**Matched Feature:** {feat}")
            if mode == "AI-Enhanced" and justify:
                st.markdown(f"*Justification:* {justify}")
            st.markdown("---")

        pdf_bytes = create_claim_chart_pdf(mappings, mode)
        st.download_button("Download PDF", data=pdf_bytes, file_name="claim_chart.pdf", mime="application/pdf")
