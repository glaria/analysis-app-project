# Campaign Analysis App

A local Streamlit app for measuring the impact of CRM campaigns with a
target/control setup. Upload the campaign data and the app reports the overall
uplift per KPI, finds the customer segments with the best and worst response
(including the optimal segments of continuous variables, found with no
predefined cut points, plus a classic bin view), and trains an uplift model to
score customers for the next campaign wave.

## Running the app

```
pip install -r requirements.txt
streamlit run Landing_Page.py
```

The app opens in the browser. Documentation on the expected input format is on
the landing page itself; a sample dataset is included in
`datasets/fetch_hillstrom_dataset.csv`.

## Pages

1. **Data Load** — upload a CSV/Excel file, review the inferred column roles
   (target/control flag, KPIs, segmentation fields), fix them if needed and
   process the data.
2. **Analysis Results** — overall uplift per KPI with significance tests,
   significant discrete segments, best/worst intervals of continuous variables,
   and a cross-KPI heatmap.
3. **Advanced Analytics** — a LightGBM uplift model per KPI: best/worst
   subgroups as readable rules, honest evaluation on a 30% holdout (Qini curve),
   and downloads of the scored customer base and a Top-25% targeting list.

Processed data is handed between pages through `pages/temp/` (git-ignored, may
contain real customer data).
