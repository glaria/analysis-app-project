import os
import re
import pandas as pd
import numpy as np
import math
import base64
import scipy.special as scsp
import streamlit as st
from sklearn.tree import _tree
from matplotlib import pyplot as plt


significance_treshold = 0.05


###*** Processed-data handoff between pages ***###
# Parquet is the primary format (faster, preserves dtypes); the CSV pair is kept
# as a fallback so datasets processed by older versions of the app still load.
DATA_PARQUET = "pages/temp/uploaded_data.parquet"
INFO_PARQUET = "pages/temp/user_defined_info_dataset.parquet"
DATA_CSV = "pages/temp/uploaded_data.csv"
INFO_CSV = "pages/temp/user_defined_info_dataset.csv"

def _use_parquet():
    if not os.path.exists(DATA_PARQUET):
        return False
    if not os.path.exists(DATA_CSV):
        return True
    return os.path.getmtime(DATA_PARQUET) >= os.path.getmtime(DATA_CSV)

def data_token():
    """Cache key that changes whenever the Data Load page writes new data."""
    return os.path.getmtime(DATA_PARQUET if _use_parquet() else DATA_CSV)

@st.cache_data
def load_processed_data(token):
    """Load the dataset and metadata written by the Data Load page.
    `token` (see data_token) invalidates the cache when the files change.
    The TGCG column is already lowercased here for all consumer pages."""
    if _use_parquet():
        dataset = pd.read_parquet(DATA_PARQUET)
        information_dataset = pd.read_parquet(INFO_PARQUET)
    else:
        dataset = pd.read_csv(DATA_CSV, sep=',')
        information_dataset = pd.read_csv(INFO_CSV, sep=',')
    tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]
    dataset[tgcg_column] = dataset[tgcg_column].str.lower()
    return dataset, information_dataset
###***    ***###


def z2p(z):
    """From z-score return p-value."""
    return 2*(1- (0.5 * (1 + scsp.erf(abs(z) / math.sqrt(2)))))

def zscore(p1, p2, n1, n2): # p1, p2 proportions
    """ Obtain zscore of 2 proportions sample """
    p = (p1*float(n1) + p2*float(n2))/(float(n1) + float(n2))
    numerator = p1 - p2
    denominator = math.sqrt(p*(1-p)*((1/n1)+(1/n2)))
    return numerator/denominator

def proportions_p_value(p1, p2, n1, n2):
    """P-value of a two-proportion z-test, or None when the test is undefined
    (empty groups, or pooled proportion of exactly 0 or 1)."""
    if n1 == 0 or n2 == 0:
        return None
    pooled = (p1 * float(n1) + p2 * float(n2)) / (float(n1) + float(n2))
    if pooled <= 0 or pooled >= 1:
        return None
    return z2p(zscore(p1, p2, n1, n2))

def kadane_algorithm(input_list):
    max_current = max_global = input_list[0]
    start = end = 0
    for i in range(1, len(input_list)):
        if input_list[i] > max_current + input_list[i]:
            max_current = input_list[i]
            start = i
        else:
            max_current += input_list[i]
        if max_current > max_global:
            max_global = max_current
            end = i
    return max_global, start, end

def kadane_algorithm_mod(input_list):
    curr_sum = max_total = input_list[0]
    start = end = 0
    for i in range(1, len(input_list)):
        curr_sum += input_list[i]
        if curr_sum > max_total:
            max_total = curr_sum
            end = i
        if curr_sum < 0:
            curr_sum = 0
    curr_sum = 0
    for i in range(end, -1, -1): #second iteration for identifying the start of the interval of max sum
        curr_sum += input_list[i]
        if curr_sum == max_total:
            start = i
            break
    return max_total, start, end

def get_negative_array(values):
    transformed_list = []
    for value in values:
        if value > 0.5:
            transformed_list.append(-1)
        elif value < -0.5:
            transformed_list.append(1)
        else:
            transformed_list.append(value)
    return transformed_list


####***Functions used during dataload***####
_KPI_NAME_PATTERN = re.compile(r'kpi|convers|convert|response|accept|purchase|redeem', re.IGNORECASE)

def _infer_tgcg_column(dataset: pd.DataFrame):
    """Exact name first, otherwise any column whose values are exactly target/control."""
    if 'TGCG' in dataset.columns:
        return 'TGCG'
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            values = set(dataset[column].dropna().astype(str).str.lower().unique())
            if 0 < len(values) <= 2 and values <= {'target', 'control'}:
                return column
    return None

def _infer_pk_column(dataset: pd.DataFrame, tgcg_column):
    """Exact name first, otherwise the most-unique integer/string column
    (>= 99% distinct values). Floats are excluded: near-unique floats are
    usually measures (spend, scores), not identifiers."""
    if 'CUSTOMERNUMBER' in dataset.columns:
        return 'CUSTOMERNUMBER'
    n = len(dataset)
    if n == 0:
        return None
    pk_column, best_ratio = None, 0.99
    for column in dataset.columns:
        if column == tgcg_column:
            continue
        if pd.api.types.is_integer_dtype(dataset[column]) or dataset[column].dtype == 'object':
            ratio = dataset[column].nunique() / n
            if ratio >= best_ratio:
                pk_column, best_ratio = column, ratio
    return pk_column

def infer_datatypes_and_metatypes(dataset: pd.DataFrame) -> pd.DataFrame:
    info_data = {'COLUMN': [], 'DATATYPE': [], 'METATYPE': []}

    tgcg_column = _infer_tgcg_column(dataset)
    pk_column = _infer_pk_column(dataset, tgcg_column)

    for column in dataset.columns:
        dtype = dataset[column].dtype
        info_data['COLUMN'].append(column)

        if column == pk_column:
            info_data['DATATYPE'].append('NUMERIC' if pd.api.types.is_numeric_dtype(dataset[column]) else 'STRING')
            info_data['METATYPE'].append('PK')
        elif column == tgcg_column:
            info_data['DATATYPE'].append('STRING')
            info_data['METATYPE'].append('TGCG')
        elif dtype == 'bool':
            info_data['DATATYPE'].append('BOOL')
            info_data['METATYPE'].append('KPI')
        elif dtype == 'object':
            info_data['DATATYPE'].append('STRING')
            info_data['METATYPE'].append('SF')
        elif set(dataset[column].dropna().unique()) <= {0, 1} and _KPI_NAME_PATTERN.search(column):
            # binary numeric column whose name suggests an outcome -> KPI
            info_data['DATATYPE'].append('NUM_ST')
            info_data['METATYPE'].append('KPI')
        elif dataset[column].nunique() < 10:
            info_data['DATATYPE'].append('NUM_ST')
            info_data['METATYPE'].append('SF')
        else:
            info_data['DATATYPE'].append('NUMERIC')
            info_data['METATYPE'].append('SF')

    inferred_info_dataset = pd.DataFrame(info_data)
    return inferred_info_dataset

def _is_numeric_column(col: pd.Series) -> bool:
    """All non-null values are numeric (bools count as numeric, like isinstance(x, int))."""
    if pd.api.types.is_numeric_dtype(col):
        return True
    inferred = pd.api.types.infer_dtype(col, skipna=True)
    return inferred in ('integer', 'floating', 'mixed-integer-float', 'boolean', 'empty')

def _is_string_column(col: pd.Series) -> bool:
    return pd.api.types.infer_dtype(col, skipna=True) in ('string', 'empty')

def _is_bool_column(col: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(col):
        return True
    return pd.api.types.infer_dtype(col, skipna=True) in ('boolean', 'empty')

def validate_datatypes_and_metatypes(dataset: pd.DataFrame, info_dataset: pd.DataFrame) -> bool:
    datatype_values = ['BOOL', 'STRING', 'NUM_ST', 'NUMERIC']
    metatype_values = ['TGCG', 'PK', 'KPI', 'SF']

    for index, row in info_dataset.iterrows():
        column = row['COLUMN']
        datatype = row['DATATYPE']
        metatype = row['METATYPE']

        if datatype not in datatype_values or metatype not in metatype_values:
            return False

        col = dataset[column]

        if metatype == 'TGCG' and not col.dropna().astype(str).str.lower().isin(['target', 'control']).all():
            return False

        if datatype == 'BOOL' and not _is_bool_column(col):
            return False

        if datatype == 'STRING' and not _is_string_column(col):
            return False

        if datatype == 'NUM_ST' and not (col.nunique() < 10 and _is_numeric_column(col)):
            return False

        if datatype == 'NUMERIC' and not _is_numeric_column(col):
            return False

        if metatype == 'KPI' and not _is_numeric_column(col):
            return False

    return True
###***    ***###

def format_float(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value


def calculate_metrics(df, kpi_columns, tgcg_column):
    """Calculates metrics for a list of KPIs.
    All columns are numeric; formatting happens at display time (style_metrics)."""
    tg = df[df[tgcg_column] == 'target']
    cg = df[df[tgcg_column] == 'control']
    tg_total = len(tg)
    cg_total = len(cg)

    metrics = []
    for kpi in kpi_columns:
        tg_acceptors = tg[kpi].sum()
        tg_acceptance = round((tg_acceptors / tg_total)*100,2) if tg_total != 0 else 0

        cg_acceptors = cg[kpi].sum()
        cg_acceptance = round((cg_acceptors / cg_total) * 100, 2) if cg_total != 0 else 0

        uplift = tg_acceptance - cg_acceptance
        p_value = proportions_p_value(tg_acceptors/tg_total, cg_acceptors/cg_total, tg_total, cg_total) if tg_total != 0 and cg_total != 0 else None

        metrics.append([kpi, float(tg_acceptors), tg_acceptance, float(cg_acceptors), cg_acceptance, uplift, p_value])

    result_df = pd.DataFrame(metrics, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    result_df['P-value'] = pd.to_numeric(result_df['P-value'], errors='coerce')

    return result_df

def calculate_metrics2(subset, kpi, tgcg_column):
    """Single-KPI wrapper around calculate_metrics (kept for backwards compatibility)."""
    return calculate_metrics(subset, [kpi], tgcg_column)


def highlight_pvalue(row, threshold=None):
    """Highlights rows with P-value <= threshold (default: significance_treshold)."""
    if threshold is None:
        threshold = significance_treshold
    if float(row["P-value"]) <= threshold and float(row["Uplift (%)"]) >= 0:
        return ["background-color: #CCFFCC"] * len(row)
    elif float(row["P-value"]) <= threshold and float(row["Uplift (%)"]) < 0:
        return ["background-color: #FFEAEA"] * len(row)
    else:
        return [""] * len(row)


_METRIC_FORMATS = {
    "TG Acceptors": "{:.2f}",
    "TG Acceptance (%)": "{:.2f}",
    "CG Acceptors": "{:.2f}",
    "CG Acceptance (%)": "{:.2f}",
    "Uplift (%)": "{:.2f}",
    "P-value": "{:.4f}",
}

def style_metrics(df, threshold=None):
    """Significance highlighting + display formatting for a metrics DataFrame.
    Only float columns are formatted, so integer Acceptors columns stay integers."""
    formats = {c: f for c, f in _METRIC_FORMATS.items()
               if c in df.columns and pd.api.types.is_float_dtype(df[c])}
    return df.style.apply(highlight_pvalue, axis=1, threshold=threshold).format(formats, na_rep="")

def filter_and_display(df, pvalue_threshold, seg_column, unique_value):
    """Filters and displays the results."""
    df = df[df['P-value'] <= pvalue_threshold]
    if not df.empty:
        print(f"Segment: {seg_column} = {unique_value}")
        print(df.to_string(index=False)) # Display the dataframe without the index

def download_csv_link(df, filename, message="Click here to download this table"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{message}</a>'
    return href

###*** Functions exclusive of the Advanced Analytics page ***###
def oversample(df, group_cols):
    """Resample every (group_cols) group up to the size of the largest group."""
    max_size = df.groupby(group_cols, observed=True).size().max()
    return df.groupby(group_cols, observed=True, group_keys=False).sample(n=max_size, replace=True)

def balanced_sample_weight(df, group_cols):
    """Per-row training weights equivalent to oversampling every (group_cols)
    group to the largest group's size, without duplicating any rows."""
    group_sizes = df.groupby(group_cols, observed=True)[group_cols[0]].transform('size')
    return (group_sizes.max() / group_sizes).to_numpy()

def get_rules(tree, feature_names, class_names, class_of_interest):
    """ Extract the rules of a decisition tree algorithm """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            
            # Check if the feature is a result of one-hot encoding
            if '==' in name:
                feature, value = name.split('==')
                p1 += [f"({feature} <> {value})"]
                p2 += [f"({feature} = {value})"]
            else:
                p1 += [f"({name} <= {np.round(threshold, 2)})"]
                p2 += [f"({name} > {np.round(threshold, 2)})"]
            
            recurse(tree_.children_left[node], p1, paths)
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    paths = sorted(paths, key=lambda x: x[-1][1], reverse=True)
    # generate rules
    rules = []
    for path in paths:
        rule = ""

        for p in path[:-1]:
            if rule != "":
                rule += " and \n"
            rule += str(p)
        
        if class_names[np.argmax(path[-1][0][0])] == class_of_interest:
            rule += f"\n\n**(samples: {path[-1][1]})**"
            rules.append(rule)

    return rules

def qini_curve(y_true, uplift_score):
    # Sorting data by the uplift score
    data = pd.DataFrame({'y_true': y_true, 'uplift_score': uplift_score}).sort_values('uplift_score', ascending=False)
    data.reset_index(drop=True, inplace=True)
    
    data['target_cumsum'] = data.y_true.cumsum()
    data['all_cumnum'] = range(1, len(data) + 1)

    # Calculating the cumulative uplift as proportion
    data['uplift_cum'] = data['target_cumsum'] / data['all_cumnum'] - data.iloc[0]['target_cumsum'] / len(data)
    data['proportion_targeted'] = data['all_cumnum'] / len(data)  # new line to calculate proportion targeted

    # Calculating the baseline (random model)
    random_model = data['target_cumsum'].iloc[-1] / len(data) * data['proportion_targeted']

    # Creating a figure and an axis
    fig, ax = plt.subplots()

    # Drawing the Qini curve with proportion targeted on x-axis
    ax.plot(data['proportion_targeted'], data['uplift_cum'], label='Model')

    # Drawing the baseline with proportion targeted
    ax.plot(data['proportion_targeted'], random_model, label='Random')

    # Labels and legend
    ax.set_xlabel('Proportion targeted')
    ax.set_ylabel('Cumulative Uplift')
    ax.legend()

    # Calculating the Qini area
    qini_area = (data['uplift_cum'] - random_model).sum() / len(data)

    return fig, ax, qini_area



