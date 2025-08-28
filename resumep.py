import streamlit as st
import google.generativeai as genai
import PyPDF2
import pandas as pd
import io

# --- Configuration ---
# It's recommended to set your API key as a Streamlit secret.
# Go to your app's settings -> secrets and add a secret called "GEMINI_API_KEY"
try:
    # Attempt to get the API key from Streamlit secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except FileNotFoundError:
    # This block will run when running locally if no secrets file is found
    st.warning("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
    # For local development, you might use an environment variable or input field
    # For this example, we'll ask the user to input it if not found.
    GEMINI_API_KEY = st.text_input("Enter your Gemini API Key:", type="password")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"An error occurred while configuring the Gemini API: {e}")
    st.stop()


# --- Gemini Model Configuration ---
generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    Args:
        pdf_file: An uploaded PDF file object.
    Returns:
        A string containing the extracted text, or None if extraction fails.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def generate_table_from_resumes(resume_texts):
    """
    Uses the Gemini API to generate a structured table from resume texts.
    Args:
        resume_texts: A list of strings, where each string is the text of a resume.
    Returns:
        A string containing the generated table in Markdown format, or an error message.
    """
    if not resume_texts:
        return "No resume text provided."

    # Combine all resume texts into a single prompt for the model
    combined_text = "\n\n--- NEW RESUME ---\n\n".join(resume_texts)

    prompt = f"""
    Based on the following resume texts, create a Markdown table summarizing the key information for each candidate.
    The table should have the following columns: "Name", "Age / DOB", "Years of Experience", "Specialization / JD", "Key Skills / Certifications", and "Other Details".
    Extract the information as accurately as possible. If a piece of information is not found, write "Not specified".

    Resume Texts:
    {combined_text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"


# --- Streamlit App UI ---

st.set_page_config(page_title="Gemini Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 Gemini Resume Analyzer")
st.markdown("Upload multiple PDF resumes to generate a comparative table of candidate details.")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

if uploaded_files:
    if st.button("✨ Generate Analysis", type="primary"):
        with st.spinner("Reading PDFs and analyzing with Gemini... Please wait."):
            # 1. Extract text from each uploaded PDF
            all_resume_texts = []
            for pdf in uploaded_files:
                st.info(f"Processing: `{pdf.name}`")
                # To read the file, we need to wrap it in a BytesIO object
                file_bytes = io.BytesIO(pdf.getvalue())
                text = extract_text_from_pdf(file_bytes)
                if text:
                    all_resume_texts.append(text)
                else:
                    st.warning(f"Could not extract text from `{pdf.name}`. Skipping.")

            # 2. Generate the table using Gemini if we have text
            if all_resume_texts:
                st.success("Successfully extracted text from all PDFs. Now generating the table...")
                generated_content = generate_table_from_resumes(all_resume_texts)

                # 3. Display the result
                st.subheader("Candidate Comparison Table")
                st.markdown(generated_content)

                # Optional: Try to convert markdown table to a Pandas DataFrame for better display
                try:
                    # A simple way to convert markdown table to list of lists
                    lines = generated_content.strip().split('\n')
                    header = [h.strip() for h in lines[0].strip('|').split('|')]
                    data = []
                    for line in lines[2:]: # Skip header and separator
                        rows = [r.strip() for r in line.strip('|').split('|')]
                        data.append(rows)

                    df = pd.DataFrame(data, columns=header)
                    st.subheader("Formatted Data Table")
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.info("Could not automatically convert the output to a formatted table, displaying raw output above.")


else:
    st.info("Please upload one or more PDF files to get started.")

st.markdown("---")
st.markdown("Powered by Google Gemini")
