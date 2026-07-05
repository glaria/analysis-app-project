import streamlit as st
import plotly.express as px
from app_functions import *
import itertools
import plotly.graph_objects as go


st.set_page_config(page_title="Analysis", page_icon="📈", layout="wide",)

st.markdown("# Campaign results")

# dataset and information_dataset come from the cached loader: reruns of this
# page reuse the in-memory copy instead of re-reading the files from disk.
# The token changes whenever the Data Load page writes new data, which
# invalidates every cached computation below.
try:
    token = data_token()
except (FileNotFoundError, OSError):
    st.info("No processed data found. Please upload and process a dataset in the Data Load page first.")
    st.markdown("Go to the [Data Load page](Data_Load)")
    st.stop()

dataset, information_dataset = load_processed_data(token)

# Analysis controls
st.sidebar.header("Analysis settings")
significance_threshold = st.sidebar.slider(
    "Significance threshold (p-value)", min_value=0.01, max_value=0.20,
    value=significance_treshold, step=0.01)
min_group_size = st.sidebar.number_input(
    "Minimum records per group (target and control) to test a segment",
    min_value=0, value=50, step=10)

# Extract the TGCG column from the dataset (values already lowercased by the loader)
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

tgcg_counts = dataset[tgcg_column].value_counts()

# Create a pie chart with counts and percentages
fig = px.pie(tgcg_counts.reset_index(), values=tgcg_counts, names = tgcg_column, title="Target vs Control Groups")

# Display the total counts and percentages in the labels
fig.update_traces(textinfo='label+percent', textfont_size=12, insidetextorientation='radial')

# Show the pie chart in the Streamlit app
st.plotly_chart(fig)

# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values


@st.cache_data
def overall_results(token):
    dataset, information_dataset = load_processed_data(token)
    tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]
    kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values
    return calculate_metrics(dataset, kpi_columns, tgcg_column)


# Calculate metrics for each KPI
result_df = overall_results(token)

st.write(style_metrics(result_df, significance_threshold))

# Add a download button for CSV
st.download_button("Download this table", result_df.to_csv(index=False).encode('utf-8'),
                   file_name="results.csv", mime="text/csv")

# Segment fields
st.markdown(f"# Discrete Segments with significant results")


@st.cache_data
def discrete_segment_results(token, min_size):
    """All (segment value, KPI) combinations and the number of segment values
    skipped for having fewer than min_size target or control records.
    The p-value filter is applied outside this function, so moving the
    significance slider does not re-run the scan."""
    dataset, information_dataset = load_processed_data(token)
    tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]
    kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values
    segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') &
                                                  (information_dataset['DATATYPE'].isin(['NUM_ST', 'BOOL', 'STRING'])),
                                                  'COLUMN'].values

    all_results = []
    skipped = 0
    for seg_column in segmentation_columns:
        for unique_value in dataset[seg_column].unique():
            subset = dataset[dataset[seg_column] == unique_value]

            # the z-test is unreliable on tiny groups: skip segments below min_size
            group_counts = subset[tgcg_column].value_counts()
            if group_counts.get('target', 0) < min_size or group_counts.get('control', 0) < min_size:
                skipped += 1
                continue

            result_df = calculate_metrics(subset, kpi_columns, tgcg_column)
            if not result_df.empty:
                # Add 'Segmentation Column' to result_df
                result_df.insert(0, 'Segmentation Column', seg_column)
                # Add 'value' column to result_df
                result_df.insert(1, 'value', unique_value)
                # Add result_df to all_results list
                all_results.append(result_df)

    if not all_results:
        return None, skipped

    all_results_df = pd.concat(all_results)
    # Reset the DataFrame's index to ensure it is unique
    all_results_df.reset_index(drop=True, inplace=True)
    return all_results_df, skipped


all_results_df, skipped_segments = discrete_segment_results(token, min_group_size)
if all_results_df is not None:
    # keep only the significant rows (cheap: the scan itself is cached above)
    significant_df = all_results_df[all_results_df['P-value'] <= significance_threshold].reset_index(drop=True)
else:
    significant_df = None
if significant_df is not None and not significant_df.empty:
    #apply the style and display de df
    st.dataframe(style_metrics(significant_df, significance_threshold))
else:
    st.info("No discrete segments with statistically significant results were found.")
if skipped_segments:
    st.caption(f"{skipped_segments} segment value(s) not tested: fewer than {min_group_size} target or control records.")


##seccion variables segmentacion continuas
#Para calcular los mejores segmentos (intervalos) de variables continuas (sin definir cuantiles ) primero debemos reescalar el control group para tener el mismo número de elementos que target


@st.cache_data
def continuous_interval_results(token, min_size):
    """Best/worst intervals of the continuous segmentation fields (Kadane).
    The p-value filter is applied outside this function, so moving the
    significance slider does not re-run the scan."""
    dataset, information_dataset = load_processed_data(token)
    tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]
    kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values

    # Identify the continuous segmentation columns
    continuous_segmentation_columns = information_dataset.loc[
        (information_dataset['METATYPE'] == 'SF') &
        (information_dataset['DATATYPE'].isin(['NUMERIC'])), 'COLUMN'].values

    # Identify the minority and majority classes
    minority_class = dataset[tgcg_column].value_counts().idxmin()
    majority_class = dataset[tgcg_column].value_counts().idxmax()

    # Split the dataset into two based on the minority and majority classes
    minority_df = dataset[dataset[tgcg_column] == minority_class]
    majority_df = dataset[dataset[tgcg_column] == majority_class]

    # Oversample the minority class
    minority_oversampled = minority_df.sample(len(majority_df), replace=True, random_state=42)

    # Combine the oversampled dataframe with the majority class dataframe
    oversampled_df = pd.concat([majority_df, minority_oversampled], axis=0)

    result_columns = ["Segmentation Field", "Lower limit", "Upper limit", "KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"]
    # Collect result rows in a list and build the DataFrame once at the end
    result_rows = []

    def process_segmentation(granular_uplift_array, concatenated_df, dataset, seg_column, kpi, tgcg_column):
        max_granular_uplift, start_index, end_index = kadane_algorithm(granular_uplift_array)
        if start_index > end_index:
            max_granular_uplift, start_index, end_index = kadane_algorithm_mod(granular_uplift_array)

        start_value = concatenated_df.iloc[start_index][seg_column]
        end_value = concatenated_df.iloc[end_index][seg_column]

        filtered_dataset = dataset[(dataset[seg_column] >= start_value) & (dataset[seg_column] <= end_value)]

        # the z-test is unreliable on tiny groups: skip intervals below min_size
        group_counts = filtered_dataset[tgcg_column].value_counts()
        if group_counts.get('target', 0) < min_size or group_counts.get('control', 0) < min_size:
            return

        result_df = calculate_metrics2(filtered_dataset, kpi, tgcg_column)

        for index, row in result_df.iterrows():
            result_rows.append({
                "Segmentation Field": seg_column,
                "Lower limit": start_value,
                "Upper limit": end_value,
                "KPI": kpi,
                "TG Acceptors": row["TG Acceptors"],
                "TG Acceptance (%)": row["TG Acceptance (%)"],
                "CG Acceptors": row["CG Acceptors"],
                "CG Acceptance (%)": row["CG Acceptance (%)"],
                "Uplift (%)": row["Uplift (%)"],
                "P-value": row["P-value"],
            })

    # Iterate over the different numerical fields for which we want to calculate the best intervals
    for seg_column in continuous_segmentation_columns:
        for kpi in kpi_columns:
            # Create a dataframe with 'target' records only
            target_df = oversampled_df[oversampled_df[tgcg_column] == 'target'][[tgcg_column, seg_column,kpi]]

            #calculate target acceptance (to be used as a penalty in the Kadane's algorithm )
            target_acceptance = target_df[kpi].sum()/len(target_df)

            # Create a dataframe with 'control' records only
            control_df = oversampled_df[oversampled_df[tgcg_column] == 'control'][[tgcg_column, seg_column,kpi]].copy()

            #this penalty is added to avoid segments with a lot of zeroes in the kadane algorithm
            control_df[kpi] = control_df[kpi] +target_acceptance

            # Sort the dataframes
            target_df = target_df.sort_values(by=seg_column)
            control_df = control_df.sort_values(by=seg_column)

            # Reset the indices of the dataframes
            target_df.reset_index(drop=True, inplace=True)
            control_df.reset_index(drop=True, inplace=True)

            # Rename the columns in control_df before concatenating
            control_df.columns = [col + "_control" for col in control_df.columns]

            # Concatenate the two dataframes
            concatenated_df = pd.concat([target_df, control_df], axis=1)

            # Calculate the mean of seg_column and seg_column_control, in case some values do not match exactly
            concatenated_df[seg_column] = (concatenated_df[seg_column] + concatenated_df[seg_column + "_control"]) / 2

            # Calculate the difference between kpi and kpi_control
            concatenated_df['granular_uplift'] = concatenated_df[kpi] - concatenated_df[kpi + "_control"]

            # Find the subintervals that maximize the sum of granular_uplift
            granular_uplift_array = concatenated_df['granular_uplift'].values

            #now we calculate the best intervals for the array in both directions (max and min)
            for k in range(2):
                if k == 1:
                    granular_uplift_array = get_negative_array(granular_uplift_array)

                process_segmentation(granular_uplift_array, concatenated_df, dataset, seg_column, kpi, tgcg_column)

    results_df = pd.DataFrame(result_rows, columns=result_columns)
    if results_df.empty:
        return results_df

    results_df["TG Acceptors"] = results_df["TG Acceptors"].round().astype(int)
    results_df["CG Acceptors"] = results_df["CG Acceptors"].round().astype(int)
    return results_df


results_df = continuous_interval_results(token, min_group_size)
#exclude non-significant results (cheap: the scan itself is cached above)
if not results_df.empty:
    results_df = results_df[results_df['P-value'] <= significance_threshold].reset_index(drop=True)

st.markdown(f"# Continuous variables with significant results")


if not results_df.empty:
    st.dataframe(style_metrics(results_df, significance_threshold))
else:
    st.info("No continuous-variable intervals with statistically significant results were found.")

##fin seccion variables segmentacion continuas

st.markdown(f"# Cross-KPI results")


@st.cache_data
def cross_kpi_matrix(token):
    """Relative uplift for every pair of KPI outcomes.
    One groupby per KPI pair instead of one full-dataframe scan per matrix cell."""
    dataset, information_dataset = load_processed_data(token)
    tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]
    kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values

    kpi_labels = [f'{kpi} = {val}' for kpi in kpi_columns for val in [0, 1]]
    matrix_df = pd.DataFrame(index=kpi_labels, columns=kpi_labels)

    tg_total = (dataset[tgcg_column] == 'target').sum()
    cg_total = (dataset[tgcg_column] == 'control').sum()

    for kpi1, kpi2 in itertools.product(kpi_columns, repeat=2):
        if kpi1 == kpi2:
            counts = dataset.groupby([tgcg_column, kpi1], observed=True).size()
            def get_count(group, value1, value2):
                return counts.get((group, value1), 0) if value1 == value2 else 0
        else:
            counts = dataset.groupby([tgcg_column, kpi1, kpi2], observed=True).size()
            def get_count(group, value1, value2):
                return counts.get((group, value1, value2), 0)

        for value1, value2 in itertools.product([0, 1], repeat=2):
            tg_count = get_count('target', value1, value2)
            cg_count = get_count('control', value1, value2)

            tg_acceptance = round((tg_count / tg_total) * 100, 2) if tg_total != 0 else 0
            cg_acceptance = round((cg_count / cg_total) * 100, 2) if cg_total != 0 else 0
            uplift = tg_acceptance - cg_acceptance
            relative_uplift = (uplift / cg_acceptance) * 100 if cg_acceptance != 0 else 0
            matrix_df.loc[f'{kpi1} = {value1}', f'{kpi2} = {value2}'] = relative_uplift

    return matrix_df


matrix_df = cross_kpi_matrix(token)

# Display the matrix in the user interface
st.write(matrix_df)

# Define colors for the heatmap: green for positive values, red for negative values.
colors = ['red', 'lightgray', 'green']

# First we convert the values to numeric, forcing non-numerics to NaN
values = pd.to_numeric(matrix_df.values.flatten(), errors='coerce').reshape(matrix_df.values.shape)

# Now we replace NaNs with 0
values = np.where(np.isnan(values), 0, values)

# Convert numeric values to strings with two decimal places
text_values = np.round(values, 0).astype(int).astype(str)
# Append '%' to each individual item
text = [f"{val}%" for val in text_values.flatten()]
# Reshape the array
text = np.array(text).reshape(values.shape)

fig = go.Figure(data=go.Heatmap(
    z=values,
    x=matrix_df.columns,
    y=matrix_df.index[::-1],  # Reverse the order of the index for the heatmap
    colorscale=colors,
    zmid=0,
    text=text,
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps = False
))

# Adjust the layout of the figure.
fig.update_layout(
    title='Heatmap of Relative Uplift',
    xaxis_nticks=len(matrix_df.columns),
    yaxis_nticks=len(matrix_df.index)
)

# Display the figure in the Streamlit user interface.
st.plotly_chart(fig)
