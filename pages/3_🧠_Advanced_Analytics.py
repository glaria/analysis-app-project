import streamlit as st
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from app_functions import *

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


@st.cache_data(show_spinner="Training uplift model...")
def advanced_analytics_for_kpi(token, kpi_original):
    """Trains the uplift model for one KPI and returns the subgroup rules and
    their measured uplift. Cached: reruns of the page reuse the results."""
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
    model = LGBMRegressor(max_depth = 5, n_estimators = 50,  min_child_samples =50)
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

    ### *** Explanining the lgbm using a simple decision tree *** ###

    # Get the feature importance as an array
    importance = model.feature_importances_

    # Create a DataFrame to represent feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,  # Feature names
        'Importance': importance  # Feature importance
    })

    # we can sort the DataFrame by feature importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Identify the top 7 most important features
    top7_features = feature_importance_df['Feature'].head(7)

    # Remove rows where top_bottom is NaN (i.e., predictions not in the top or bottom quartile)
    dataset_binary = dataset.dropna(subset=['top_bottom'])

    # Define the features (X) and the target variable (y)
    X_binary = dataset_binary[top7_features]
    y_binary = dataset_binary['top_bottom']

    # Perform one-hot encoding on the categorical features
    X_binary_encoded = pd.get_dummies(X_binary, prefix_sep='==')

    # Train the decision tree with only the top 7 most important features
    dt_model = DecisionTreeClassifier(max_depth=4)  # fix max depth
    dt_model.fit(X_binary_encoded, y_binary)

    #explain tree rules
    # Use the function get_rules to extract dt rules
    rules_top25 = get_rules(dt_model, X_binary_encoded.columns, dt_model.classes_, 'Top25%')
    rules_Bottom25 = get_rules(dt_model, X_binary_encoded.columns, dt_model.classes_, 'Bottom25%')

    #now we want to measure the uplift of the subset defined by the dt (for both bottom and top)
    # Select the same features as used for the model and apply the same transformations
    X_original = dataset[top7_features]
    X_original_encoded = pd.get_dummies(X_original, prefix_sep='==')
    # Align to the training columns: category values that only occur in the middle
    # 50% would otherwise produce a feature mismatch at prediction time
    X_original_encoded = X_original_encoded.reindex(columns=X_binary_encoded.columns, fill_value=0)

    # Use the trained model to predict the classes
    dataset['dt_classification'] = dt_model.predict(X_original_encoded)

    # Collect result rows in a list and build the DataFrame once at the end
    result_rows = []
    for value in ['Top25%','Bottom25%']:
        filtered_dataset = dataset.loc[dataset['dt_classification'] == value]
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

    return rules_top25, rules_Bottom25, results_df


if num_records > 2000000:
    st.write("The number of total records is greater than 2,000,000. The model cannot be processed.")
elif num_records < 200:
    st.write("The number of total records is less than 200. The model cannot be processed.")
elif num_control_records < 100:
    st.write("There are fewer than 100 control records. The model cannot be processed.")
elif num_target_records < 100:
    st.write("There are fewer than 100 target records. The model cannot be processed.")
else:
    for kpi in kpi_columns:
        st.header(f"KPI: {kpi}")

        rules_top25, rules_Bottom25, results_df = advanced_analytics_for_kpi(token, kpi)

        # Display the rules in Streamlit with modified background color
        st.markdown("\n\n**Best subgroups identified**")
        for rule in rules_top25:
            st.markdown(f"<div class='rules-box-top'>{rule}</div>", unsafe_allow_html=True)

        st.markdown("\n\n**Worst subgroups identified**")
        for rule in rules_Bottom25:
            st.markdown(f"<div class='rules-box-bottom'>{rule}</div>", unsafe_allow_html=True)

        st.markdown(f"**Results on the best and worst subgroups**")
        st.dataframe(results_df.style.apply(highlight_pvalue, axis=1))
