import streamlit as st

st.set_page_config(
    page_title="Campaign Analysis App",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Campaign Analysis App")
st.markdown(
    "Measure whether a CRM campaign actually worked. Upload the campaign data — "
    "customers split into a **target group** (received the campaign) and a "
    "**control group** (didn't) — and get the overall uplift, the customer "
    "segments that responded best and worst, and the customers to contact in "
    "the next wave."
)

# The three steps of the workflow, in order, each linking to its page
step1, step2, step3 = st.columns(3)

with step1, st.container(border=True):
    st.subheader("1 · Data load")
    st.markdown(
        "Upload your CSV or Excel file. The app guesses the role of each "
        "column and lets you correct it in an editable table. When everything "
        "is consistent, press **Process Data**."
    )
    st.page_link("pages/1_📊_Data_Load.py", label="Open Data Load", icon="📊")

with step2, st.container(border=True):
    st.subheader("2 · Campaign results")
    st.markdown(
        "The overall uplift per KPI, the discrete segments with significant "
        "results, the continuous variables as optimal segments and as bins, "
        "and a cross-KPI heatmap."
    )
    st.page_link("pages/2_📈_Analysis_Results.py", label="Open Analysis Results", icon="📈")

with step3, st.container(border=True):
    st.subheader("3 · Advanced analytics")
    st.markdown(
        "An uplift model per KPI: best and worst subgroups as readable rules, "
        "honest evaluation on a 30% holdout, and a scored customer base with "
        "a Top-25% targeting export."
    )
    st.page_link("pages/3_🧠_Advanced_Analytics.py", label="Open Advanced Analytics", icon="🧠")


@st.cache_data
def sample_dataset_bytes():
    with open("datasets/fetch_hillstrom_dataset.csv", "rb") as f:
        return f.read()


st.subheader("Try it without your own data")
st.markdown(
    "Download the sample dataset (the classic Hillstrom e-mail campaign) and "
    "load it in the Data Load page."
)
st.download_button(
    "Download the Hillstrom sample dataset",
    sample_dataset_bytes(),
    file_name="fetch_hillstrom_dataset.csv",
    mime="text/csv",
)

with st.expander("What your file needs"):
    st.markdown(
        """
    One row per customer. Every column gets a **meta type**:

    | Meta type | Meaning | Requirements |
    | --- | --- | --- |
    | **TGCG** | Target/control flag | Exactly one column, values `target` / `control` |
    | **KPI** | Campaign outcome to measure | At least one column, binary 0/1 |
    | **PK** | Unique customer id | Optional, at most one; enables the targeting export |
    | **SF** | Segmentation field | Everything else: the customer attributes used to find segments |

    and a **data type**: `BOOL`, `STRING`, `NUMERIC`, or `NUM_ST` (numeric with
    fewer than 10 distinct values, treated as discrete). Missing values are filled
    automatically (`NONE` for text, `0` for numbers).
    """
    )
