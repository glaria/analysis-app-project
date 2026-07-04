import streamlit as st
from matplotlib import pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from app_functions import *

# This page merges the former Advanced Analytics and Modelling pages.
# It is organised in two clearly separated sections per KPI:
#   1. Best & worst segments  - human-readable subgroups extracted from the model
#   2. Uplift model (reusable) - the model itself: honest holdout evaluation,
#      Qini curve, feature importance, scored dataset and targeting export.
# The model is evaluated on a 30% holdout that it never saw during training,
# so the reported subgroup uplifts are not inflated by overfitting.

HOLDOUT_FRACTION = 0.3

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
try:
    token = data_token()
except (FileNotFoundError, OSError):
    st.info("No processed data found. Please upload and process a dataset in the Data Load page first.")
    st.markdown("Go to the [Data Load page](Data_Load)")
    st.stop()

dataset, information_dataset = load_processed_data(token)

# Extract the TGCG column from the dataset (values already lowercased by the loader)
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values

# PK column (if defined) enables the targeting export
pk_condition = information_dataset['METATYPE'] == 'PK'
pk_column = information_dataset.loc[pk_condition, 'COLUMN'].values[0] if pk_condition.any() else None

# Conditions before processing
num_records = dataset.shape[0]
num_control_records = (dataset[tgcg_column] == 'control').sum()
num_target_records = (dataset[tgcg_column] == 'target').sum()


@st.cache_data(show_spinner="Training uplift model...")
def uplift_model_for_kpi(token, kpi_original):
    """Trains the uplift model for one KPI on a train split and evaluates it on
    the holdout. Returns the subgroup rules, holdout results, the scored full
    dataset and the model itself. Cached: reruns of the page reuse the results."""
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

    #create model_target column (vectorized: 1 when tgcg_fl == kpi value)
    kpi = kpi_original + '_model_target'
    dataset[kpi] = (dataset['tgcg_fl'] == dataset[kpi_original]).astype(int)

    feature_columns = list(segmentation_columns) + list(continue_segmentation_columns)

    # Stratified train/holdout split: the model is trained on the train split and
    # every reported result is measured on the holdout it never saw
    stratify_key = dataset[kpi].astype(str) + '_' + dataset['tgcg_fl'].astype(str)
    train_df, holdout_df = train_test_split(
        dataset, test_size=HOLDOUT_FRACTION, random_state=42, stratify=stratify_key)

    # Balanced training weights instead of physically oversampling: same
    # statistical effect as oversample() but without duplicating rows
    sample_weight = balanced_sample_weight(train_df, [kpi, 'tgcg_fl'])

    #train model
    X_train = train_df[feature_columns]
    y_train = train_df[kpi]

    model = LGBMRegressor(max_depth = 5, n_estimators = 100,  min_child_samples =50)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    ### *** Holdout evaluation *** ###
    holdout = holdout_df.copy()
    holdout['predictions'] = model.predict(holdout[feature_columns])

    # Define the 25th and 75th percentiles on the holdout scores
    upper_quartile = np.percentile(holdout['predictions'], 75)
    lower_quartile = np.percentile(holdout['predictions'], 25)

    # np.select with default=None keeps the middle 50% as real NaN
    holdout['top_bottom'] = np.select(
        [holdout['predictions'] >= upper_quartile, holdout['predictions'] <= lower_quartile],
        ['Top25%', 'Bottom25%'],
        default=None
    )

    # Uplift of the score quartiles, measured on the holdout
    quartile_rows = []
    for value in ['Top25%','Bottom25%']:
        result_df = calculate_metrics2(holdout.loc[holdout['top_bottom'] == value], kpi_original, tgcg_column)
        for index, row in result_df.iterrows():
            quartile_rows.append({"Group": value, **row.to_dict()})
    quartile_results_df = pd.DataFrame(quartile_rows, columns=["Group", "KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    if not quartile_results_df.empty:
        quartile_results_df["TG Acceptors"] = quartile_results_df["TG Acceptors"].round().astype(int)
        quartile_results_df["CG Acceptors"] = quartile_results_df["CG Acceptors"].round().astype(int)

    ### *** Explanining the lgbm using a simple decision tree *** ###

    # Feature importance of the LGBM model
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Identify the top 7 most important features
    top7_features = feature_importance_df['Feature'].head(7)

    # Remove rows where top_bottom is NaN (i.e., predictions not in the top or bottom quartile)
    holdout_binary = holdout.dropna(subset=['top_bottom'])

    # Perform one-hot encoding on the categorical features
    X_binary_encoded = pd.get_dummies(holdout_binary[top7_features], prefix_sep='==')
    y_binary = holdout_binary['top_bottom']

    # Train the decision tree with only the top 7 most important features
    dt_model = DecisionTreeClassifier(max_depth=4)  # fix max depth
    dt_model.fit(X_binary_encoded, y_binary)

    #explain tree rules
    rules_top25 = get_rules(dt_model, X_binary_encoded.columns, dt_model.classes_, 'Top25%')
    rules_Bottom25 = get_rules(dt_model, X_binary_encoded.columns, dt_model.classes_, 'Bottom25%')

    #measure the uplift of the subgroups defined by the dt on the holdout
    X_holdout_encoded = pd.get_dummies(holdout[top7_features], prefix_sep='==')
    # Align to the training columns: category values that only occur in the middle
    # 50% would otherwise produce a feature mismatch at prediction time
    X_holdout_encoded = X_holdout_encoded.reindex(columns=X_binary_encoded.columns, fill_value=0)
    holdout['dt_classification'] = dt_model.predict(X_holdout_encoded)

    subgroup_rows = []
    for value in ['Top25%','Bottom25%']:
        result_df = calculate_metrics2(holdout.loc[holdout['dt_classification'] == value], kpi_original, tgcg_column)
        for index, row in result_df.iterrows():
            subgroup_rows.append({"Group": value, **row.to_dict()})
    subgroup_results_df = pd.DataFrame(subgroup_rows, columns=["Group", "KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    if not subgroup_results_df.empty:
        subgroup_results_df["TG Acceptors"] = subgroup_results_df["TG Acceptors"].round().astype(int)
        subgroup_results_df["CG Acceptors"] = subgroup_results_df["CG Acceptors"].round().astype(int)

    ### *** Score the full dataset (for the preview, download and targeting export) *** ###
    dataset['predictions'] = model.predict(dataset[feature_columns])
    dataset['top_bottom'] = np.select(
        [dataset['predictions'] >= upper_quartile, dataset['predictions'] <= lower_quartile],
        ['Top25%', 'Bottom25%'],
        default=None
    )
    scored_dataset = dataset.drop(columns=['tgcg_fl', kpi])

    # Inputs for the Qini curve, computed on the holdout only
    qini_y_true = holdout[kpi_original].to_numpy()
    qini_scores = holdout['predictions'].to_numpy()

    return (rules_top25, rules_Bottom25, subgroup_results_df, quartile_results_df,
            scored_dataset, model, qini_y_true, qini_scores)


@st.cache_data
def scored_csv_for_kpi(token, kpi_original):
    """CSV bytes of the scored dataset, built only when the user asks for it."""
    scored_dataset = uplift_model_for_kpi(token, kpi_original)[4]
    return scored_dataset.drop(columns=['top_bottom']).to_csv(index=False).encode('utf-8')


@st.cache_data
def targeting_csv_for_kpi(token, kpi_original, pk_column):
    """CSV bytes with the PK values of the Top25% scored customers."""
    scored_dataset = uplift_model_for_kpi(token, kpi_original)[4]
    top_customers = scored_dataset.loc[scored_dataset['top_bottom'] == 'Top25%', [pk_column, 'predictions']]
    top_customers = top_customers.sort_values('predictions', ascending=False)
    return top_customers.to_csv(index=False).encode('utf-8')


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

        (rules_top25, rules_Bottom25, subgroup_results_df, quartile_results_df,
         scored_dataset, model, qini_y_true, qini_scores) = uplift_model_for_kpi(token, kpi)

        ## Section 1: human-readable segments extracted from the model ##
        st.subheader("1. Best & worst segments")
        st.caption(f"Subgroups extracted from the model. Their uplift is measured on the {HOLDOUT_FRACTION:.0%} holdout the model never saw during training.")

        # Display the rules in Streamlit with modified background color
        st.markdown("\n\n**Best subgroups identified**")
        for rule in rules_top25:
            st.markdown(f"<div class='rules-box-top'>{rule}</div>", unsafe_allow_html=True)

        st.markdown("\n\n**Worst subgroups identified**")
        for rule in rules_Bottom25:
            st.markdown(f"<div class='rules-box-bottom'>{rule}</div>", unsafe_allow_html=True)

        st.markdown(f"**Results on the best and worst subgroups (holdout)**")
        st.dataframe(style_metrics(subgroup_results_df))

        st.divider()

        ## Section 2: the uplift model itself, reusable for targeting ##
        st.subheader("2. Uplift model (reusable)")
        st.caption("The trained model, its honest evaluation on the holdout, and the scored customer base for targeting the next campaign wave.")

        st.markdown("**Uplift of the score quartiles (holdout)**")
        st.dataframe(style_metrics(quartile_results_df))

        # Qini curve on the holdout: model quality at a glance
        qini_fig, qini_ax, qini_area = qini_curve(qini_y_true, qini_scores)
        qini_ax.set_title(f"Qini curve on the holdout ({kpi})")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(qini_fig)
            st.caption(f"Qini area: {qini_area:.5f} (higher is better; 0 = no better than random targeting)")
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            lgb.plot_importance(model, ax=ax)
            ax.set_title(f"Feature importance of the kpi {kpi}")
            st.pyplot(fig)

        st.markdown("**Scored customers**")
        # Only the first rows are rendered: sending the full scored dataset to
        # the browser freezes the page at this data size
        st.dataframe(scored_dataset.drop(columns=['top_bottom']).head(1000))
        st.caption(f"Showing the first 1,000 of {len(scored_dataset):,} scored rows. Use the downloads below for the full data.")

        #downloads are built on demand and served as bytes
        if pk_column is not None:
            if st.checkbox(f"Prepare targeting list: Top 25% customers by uplift score ({kpi})", key=f"prep_targeting_{kpi}"):
                st.download_button(
                    label=f"Download targeting_top25_{kpi}.csv",
                    data=targeting_csv_for_kpi(token, kpi, pk_column),
                    file_name=f"targeting_top25_{kpi}.csv",
                    mime="text/csv",
                    key=f"targeting_{kpi}",
                )
        else:
            st.caption("Define a PK column in the Data Load page to enable the Top 25% targeting export.")

        if st.checkbox(f"Prepare full scored dataset download ({kpi})", key=f"prep_download_{kpi}"):
            st.download_button(
                label=f"Download model_kpi_{kpi}.csv",
                data=scored_csv_for_kpi(token, kpi),
                file_name=f"model_kpi_{kpi}.csv",
                mime="text/csv",
                key=f"download_{kpi}",
            )
