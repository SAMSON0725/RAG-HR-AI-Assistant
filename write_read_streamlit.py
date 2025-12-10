import os
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

st.set_page_config(page_title="HR AI Assitant Chatbot 🤖", layout="wide")

# Config
DOCS_DIR = Path(r"c:\Users\samso\Downloads\Python\Python Agents-\data").resolve()
DOCS_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = DOCS_DIR / "onboarding.xlsx"
COLUMNS = ["Name", "Bill Rate (USD)", "Location", "Years of Experience"]

# Bill rate calculation (simple tiered formula - adjust as needed)
def calculate_bill_rate(years: float) -> float:
    base = 50.0
    per_year = 10.0
    return round(base + per_year * years, 2)

# Ensure template exists
def ensure_template():
    if not EXCEL_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")

def load_table() -> pd.DataFrame:
    ensure_template()
    return pd.read_excel(EXCEL_PATH, engine="openpyxl")

def append_row(name: str, bill_rate: float, location: str, years: float) -> int:
    df = load_table()
    new = {
        "Name": name,
        "Bill Rate (USD)": bill_rate,
        "Location": location,
        "Years of Experience": years,
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
    return len(df)

def replace_table_from_bytes(excel_bytes: bytes) -> None:
    buf = BytesIO(excel_bytes)
    df = pd.read_excel(buf, engine="openpyxl")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file missing columns: {missing}")
    df = df[COLUMNS]
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")

def get_excel_bytes() -> bytes:
    ensure_template()
    with open(EXCEL_PATH, "rb") as f:
        return f.read()

# Streamlit UI
st.title("HR AI Assistant Chatbot")
st.markdown("Use the form to onboard resources or upload/download the onboarding Excel file.")

# Sidebar: upload / download / preview
with st.sidebar:
    st.header("Onboarding File")
    uploaded = st.file_uploader("Upload onboarding Excel (will replace)", type=["xlsx"])
    if uploaded is not None:
        try:
            replace_table_from_bytes(uploaded.read())
            st.success("Onboarding Excel replaced.")
        except Exception as e:
            st.error(f"Upload error: {e}")

    if st.button("Download onboarding Excel"):
        try:
            st.download_button(
                "Download Excel",
                data=get_excel_bytes(),
                file_name="onboarding.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Download error: {e}")

    if st.button("Show onboarding preview"):
        try:
            st.dataframe(load_table().head(50))
        except Exception as e:
            st.error(f"Cannot read onboarding file: {e}")

# Main: onboarding form and chat-like area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Quick Onboard")
    with st.form("onboard_form"):
        name = st.text_input("Name")
        location = st.text_input("Location")
        years = st.text_input("Years of Experience")
        submit = st.form_submit_button("Add to Onboarding Excel")
        if submit:
            try:
                if not name or not location or not years:
                    st.error("Name, Location and Years are required.")
                else:
                    y = float(years)
                    if y < 0:
                        raise ValueError("Years must be non-negative")
                    bill = calculate_bill_rate(y)
                    count = append_row(name.strip(), bill, location.strip(), y)
                    st.success(f"Added {name} — Bill Rate ${bill:.2f}. Total rows: {count}")
                    st.download_button(
                        "Download onboarding.xlsx",
                        data=get_excel_bytes(),
                        file_name="onboarding.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Error adding row: {e}")

with col2:
    st.subheader("Onboarding Helpers")
    if st.button("Show last 10 entries"):
        try:
            df = load_table()
            st.dataframe(df.tail(10))
        except Exception as e:
            st.error(f"Error reading file: {e}")

st.markdown("---")
#st.subheader("Chat (simple guidance)")
#st.markdown(
   # "Type commands or questions. For onboarding, use the form above. "
   # "This app reads/writes onboarding.xlsx in the data folder."
#)

# Footer: small preview
st.markdown("### Onboarding table preview")
try:
    st.dataframe(load_table().tail(20))
except Exception as e:
    st.error(f"Cannot read onboarding table: {e}")