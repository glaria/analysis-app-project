import streamlit as st

st.set_page_config(
    page_title="Campaign Analysis App",
    page_icon="👋",
)

st.write("# Campaign Analysis App 👋")

st.markdown(
    """
    This app measures whether a CRM campaign actually worked. You upload the
    campaign data — customers split into a **target group** (received the campaign)
    and a **control group** (didn't) — and the app tells you the overall uplift,
    which customer segments responded best and worst, and which customers to
    contact in the next wave.

    ### How to use it

    Work through the pages in the sidebar, in order:

    **1. 📊 Data Load** — Upload your CSV or Excel file. The app guesses the role
    of each column and lets you correct it in the sidebar. When everything is
    consistent, press **Process Data**.

    **2. 📈 Analysis Results** — The campaign results: overall uplift per KPI,
    discrete segments with significant results, the continuous variables analysed
    both as optimal best/worst segments and as bins (side by side, easy to
    compare), and a cross-KPI heatmap. Rows highlighted in green are
    significant positive results; red means significant negative. You can adjust
    the significance threshold and the minimum group size in the sidebar.

    **3. 🧠 Advanced Analytics** — An uplift model trained per KPI. It extracts
    the best and worst subgroups as human-readable rules, evaluates the model
    honestly on a 30% holdout (Qini curve, score quartiles), and lets you download
    the scored customer base and a Top-25% targeting list for the next campaign wave.

    ### What your file needs

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

    Want to try it without your own data? There is a sample dataset in
    `datasets/fetch_hillstrom_dataset.csv` (the classic Hillstrom e-mail campaign).

    👈 **Start with the Data Load page.**
"""
)

st.markdown("Go to the [Data Load page](Data_Load)")
