import streamlit as st
from matplotlib import pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from app_functions import *
from sklearn.tree import _tree

# TODO: define training/test sets before training the model
# TODO: gridsearch on hyperparameters

# CSS to change background color
st.markdown(
    """
    <style>
    .rules-box-top {
        background-color: #e6ffe6;  # light green
        border-radius: 5px;
        padding: 10px;
    }
    .rules-box-bottom {
        background-color: #ffe6e6;  # light red
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# dataset and information_dataset come from the cached loader; the token
# invalidates every cached computation when the Data Load page writes new data
token = data_token()
dataset, information_dataset = load_processed_data(token)

# Extract the TGCG column from the dataset (values already lowercased by the loader)
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values

# Conditions before processing
num_records = dataset.shape[0]
num_control_records = (dataset[tgcg_column] == 'control').sum()
num_target_records = (dataset[tgcg_column] == 'target').sum()


@st.cache_data(show_spinner="Training model...")
def modelling_for_kpi(token, kpi_original):
    """Trains the model for one KPI and returns the scored dataset, the
    Top/Bottom quartile results and the model itself. Cached per KPI."""
    dataset, information_dataset = load_processed_data(token)
    tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

    # 1. Identify the segmentation columns
    segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') &
                                                  (information_dataset['DATATYPE'].isin(['NUM_ST', 'BOOL', 'STRING'])),
                                                  'COLUMN'].values
    continue_segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') &
                                                  (information_dataset['DATATYPE'].isin(['NUMERIC'])),
                                                  'COLUMN'].values
    #convert category columns to category type in pandas
    for col in segmentation_columns:
        dataset[col] = dataset[col].astype('category')

    # Create new column with tgcg as flags
    dataset['tgcg_fl'] = np.where(dataset[tgcg_column] == 'target', 1, 0)

    dataset_copy = dataset.copy()
    #create model_target column (vectorized: 1 when tgcg_fl == kpi value)
    kpi = kpi_original + '_model_target'
    dataset_copy[kpi] = (dataset_copy['tgcg_fl'] == dataset_copy[kpi_original]).astype(int)

    # Create a list with all the columns of interest
    all_columns = list(segmentation_columns) + list(continue_segmentation_columns) + ['tgcg_fl'] + [kpi]

    # Create a new DataFrame that only contains the columns of interest
    df_subset = dataset_copy[all_columns]

    # Balanced training weights instead of physically oversampling: same
    # statistical effect as oversample() but without duplicating rows
    sample_weight = balanced_sample_weight(df_subset, [kpi, 'tgcg_fl'])

    #train model
    # Definir las características (X) y la variable objetivo (y)
    X = df_subset.drop(columns=['tgcg_fl', kpi])
    y = df_subset[kpi]

    # train model
    model = LGBMRegressor(max_depth = 5, n_estimators = 100,  min_child_samples =50)
    model.fit(X, y, sample_weight=sample_weight)

    # Create a new list of columns that only includes columns present in 'dataset'
    dataset_columns = list(segmentation_columns) + list(continue_segmentation_columns)

    # Define the features (X) for the original dataset
    X_dataset = dataset[dataset_columns]

    # Make predictions on the original dataset
    dataset['predictions'] = model.predict(X_dataset)

    # After creating the predictions column in the original dataset
    # Define the 25th and 75th percentiles
    upper_quartile = np.percentile(dataset['predictions'], 75)
    lower_quartile = np.percentile(dataset['predictions'], 25)

    # Create a new binary column where 'Top25%' indicates the prediction is above the upper quartile
    # and 'Bottom25%' indicates the prediction is below the lower quartile.
    # np.select with default=None keeps the middle 50% as real NaN (np.where would
    # coerce np.nan to the string 'nan', which dropna cannot remove)
    dataset['top_bottom'] = np.select(
        [dataset['predictions'] >= upper_quartile, dataset['predictions'] <= lower_quartile],
        ['Top25%', 'Bottom25%'],
        default=None
    )

    # Collect result rows in a list and build the DataFrame once at the end
    result_rows = []
    for value in ['Top25%','Bottom25%']:
        filtered_dataset = dataset.loc[dataset['top_bottom'] == value]
        result_df = calculate_metrics2(filtered_dataset, kpi_original, tgcg_column)
        for index, row in result_df.iterrows():
            result_rows.append({
                "KPI": kpi_original,
                "TG Acceptors": row["TG Acceptors"],
                "TG Acceptance (%)": row["TG Acceptance (%)"],
                "CG Acceptors": row["CG Acceptors"],
                "CG Acceptance (%)": row["CG Acceptance (%)"],
                "Uplift (%)": row["Uplift (%)"],
                "P-value": row["P-value"],
            })

    results_df = pd.DataFrame(result_rows, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    if not results_df.empty:
        results_df["TG Acceptors"] = results_df["TG Acceptors"].astype(float).round(0).astype(int)
        results_df["CG Acceptors"] = results_df["CG Acceptors"].astype(float).round(0).astype(int)

    scored_dataset = dataset.drop(columns=['tgcg_fl'])
    return scored_dataset, results_df, model


@st.cache_data
def scored_csv_for_kpi(token, kpi_original):
    """CSV bytes of the scored dataset, built only when the user asks for it."""
    scored_dataset, _, _ = modelling_for_kpi(token, kpi_original)
    return scored_dataset.drop(columns=['top_bottom']).to_csv(index=False).encode('utf-8')


if num_records > 600000:
    st.write("The number of total records is greater than 600,000. The model cannot be processed.")
elif num_records < 200:
    st.write("The number of total records is less than 200. The model cannot be processed.")
elif num_control_records < 100:
    st.write("There are fewer than 100 control records. The model cannot be processed.")
elif num_target_records < 100:
    st.write("There are fewer than 100 target records. The model cannot be processed.")
else:
    for kpi in kpi_columns:
        kpi_name = kpi

        scored_dataset, results_df, model = modelling_for_kpi(token, kpi)

        st.header(f"Model scores of the kpi: {kpi_name}")
        # Only the first rows are rendered: sending the full scored dataset to
        # the browser freezes the page at this data size
        st.dataframe(scored_dataset.drop(columns=['top_bottom']).head(1000))
        st.caption(f"Showing the first 1,000 of {len(scored_dataset):,} scored rows. Use the download below for the full dataset.")

        st.markdown(f"**Results model**")
        st.dataframe(results_df.style.apply(highlight_pvalue, axis=1))

        #download of the full scored dataset (built on demand, served as bytes
        #instead of being base64-embedded in the page HTML)
        if st.checkbox(f"Prepare full scored dataset download ({kpi_name})", key=f"prep_download_{kpi_name}"):
            st.download_button(
                label=f"Download model_kpi_{kpi_name}.csv",
                data=scored_csv_for_kpi(token, kpi),
                file_name=f"model_kpi_{kpi_name}.csv",
                mime="text/csv",
                key=f"download_{kpi_name}",
            )

        ### *** Feature importance of the trained model *** ###

        fig, ax = plt.subplots(figsize=(12, 10))  # Create a subplot to adjust the size if necessary
        lgb.plot_importance(model, ax=ax)  # Pass the axis to the function to plot

        ax.set_title(f"Feature importance of the kpi {kpi_name}")
        st.pyplot(fig)  # Display the figure in Streamlit
