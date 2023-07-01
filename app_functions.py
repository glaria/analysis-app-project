import pandas as pd
import numpy as np
import math
import base64
import scipy.special as scsp
from sklearn.tree import _tree

significance_treshold = 0.1


def z2p(z):
    """From z-score return p-value."""
    return 2*(1- (0.5 * (1 + scsp.erf(abs(z) / math.sqrt(2)))))

def zscore(p1, p2, n1, n2): # p1, p2 proportions
    """ Obtain zscore of 2 proportions sample """
    p = (p1*float(n1) + p2*float(n2))/(float(n1) + float(n2))
    numerator = p1 - p2
    denominator = math.sqrt(p*(1-p)*((1/n1)+(1/n2)))
    return numerator/denominator

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


####***Functions used during dataload***####
def infer_datatypes_and_metatypes(dataset: pd.DataFrame) -> pd.DataFrame:
    info_data = {'COLUMN': [], 'DATATYPE': [], 'METATYPE': []}

    for column in dataset.columns:
        dtype = dataset[column].dtype

        if column == 'CUSTOMERNUMBER':
            info_data['COLUMN'].append(column)
            info_data['DATATYPE'].append('NUMERIC')
            info_data['METATYPE'].append('PK')
        elif column == 'TGCG':
            info_data['COLUMN'].append(column)
            info_data['DATATYPE'].append('STRING')
            info_data['METATYPE'].append('TGCG')
        else:
            info_data['COLUMN'].append(column)

            if dtype == 'bool':
                info_data['DATATYPE'].append('BOOL')
                info_data['METATYPE'].append('KPI')
            elif dtype == 'object':
                info_data['DATATYPE'].append('STRING')
                info_data['METATYPE'].append('SF')
            elif dataset[column].nunique() < 10:
                info_data['DATATYPE'].append('NUM_ST')
                info_data['METATYPE'].append('SF')
            else:
                info_data['DATATYPE'].append('NUMERIC')
                info_data['METATYPE'].append('SF')

    inferred_info_dataset = pd.DataFrame(info_data)
    return inferred_info_dataset

def validate_datatypes_and_metatypes(dataset: pd.DataFrame, info_dataset: pd.DataFrame) -> bool:
    datatype_values = ['BOOL', 'STRING', 'NUM_ST', 'NUMERIC']
    metatype_values = ['TGCG', 'PK', 'KPI', 'SF']

    for index, row in info_dataset.iterrows():
        column = row['COLUMN']
        datatype = row['DATATYPE']
        metatype = row['METATYPE']

        if datatype not in datatype_values or metatype not in metatype_values:
            return False

        if metatype == 'TGCG' and not dataset[column].apply(lambda x: x.lower() in ['target', 'control'] if pd.notnull(x) else True).all():
            return False

        if datatype == 'BOOL' and not dataset[column].apply(lambda x: isinstance(x, bool) if pd.notnull(x) else True).all():
            return False

        if datatype == 'STRING' and not dataset[column].apply(lambda x: isinstance(x, str) if pd.notnull(x) else True).all():
            return False

        if datatype == 'NUM_ST' and not (dataset[column].nunique() < 10 and dataset[column].apply(lambda x: isinstance(x, (int, float)) if pd.notnull(x) else True).all()):
            return False

        if datatype == 'NUMERIC' and not dataset[column].apply(lambda x: isinstance(x, (int, float)) if pd.notnull(x) else True).all():
            return False

        if metatype == 'KPI' and not dataset[column].apply(lambda x: isinstance(x, (int, float)) if pd.notnull(x) else True).all():
            return False

    return True
###***    ***###

def format_float(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value


def calculate_metrics2(subset, kpi, tgcg_column):
    """Calculates metrics for a specific KPI."""
    metrics = []
    tg_acceptors = subset.loc[subset[tgcg_column] == 'target', kpi].sum()
    tg_total = len(subset.loc[subset[tgcg_column] == 'target'])
    tg_acceptance = round((tg_acceptors / tg_total)*100,2) if tg_total != 0 else 0

    cg_acceptors = subset.loc[subset[tgcg_column] == 'control', kpi].sum()
    cg_total = len(subset.loc[subset[tgcg_column] == 'control'])
    cg_acceptance = round((cg_acceptors / cg_total) * 100, 2) if cg_total != 0 else 0

    uplift = tg_acceptance - cg_acceptance
    p_value = z2p(zscore(float(tg_acceptors)/float(tg_total), float(cg_acceptors)/float(cg_total),float(tg_total), float(cg_total))) if tg_total != 0 and cg_total != 0 else None
    
    metrics.append([kpi, "{:.2f}".format(tg_acceptors), "{:.2f}".format(tg_acceptance), "{:.2f}".format(cg_acceptors), "{:.2f}".format(cg_acceptance), "{:.2f}".format(uplift), p_value])
    result_df = pd.DataFrame(metrics, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    result_df['P-value'] = pd.to_numeric(result_df['P-value'], errors='coerce')
    return result_df


def highlight_pvalue(row):
    """Highlights rows with P-value <= significance_treshold."""
    if float(row["P-value"]) <= significance_treshold:
        return ["background-color: lightgreen"] * len(row)
    return [""] * len(row)

def calculate_metrics(df, kpi_columns, tgcg_column):
    """Calculates metrics for a list of KPIs."""
    metrics = []
    for kpi in kpi_columns:
        tg = df[df[tgcg_column] == 'target']
        cg = df[df[tgcg_column] == 'control']

        tg_acceptors = tg[kpi].sum()
        tg_total = len(tg)
        tg_acceptance = round((tg_acceptors / tg_total)*100,2) if tg_total != 0 else 0

        cg_acceptors = cg[kpi].sum()
        cg_total = len(cg)
        cg_acceptance = round((cg_acceptors / cg_total) * 100, 2) if cg_total != 0 else 0

        uplift = tg_acceptance - cg_acceptance
        p_value = z2p(zscore(tg_acceptors/tg_total, cg_acceptors/cg_total, tg_total, cg_total)) if tg_total != 0 and cg_total != 0 else None

        metrics.append([kpi, "{:.2f}".format(tg_acceptors), "{:.2f}".format(tg_acceptance), "{:.2f}".format(cg_acceptors), "{:.2f}".format(cg_acceptance), "{:.2f}".format(uplift), p_value])

    result_df = pd.DataFrame(metrics, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    result_df['P-value'] = pd.to_numeric(result_df['P-value'], errors='coerce')

    return result_df

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
    # biggest group in terms of size
    max_size = df[group_cols].value_counts().max()
    # empty df that will be filled in
    df_oversampled = pd.DataFrame()
    # group columns per group_cols and oversample each of the groups
    for group, group_df in df.groupby(group_cols):
        oversampled_group = group_df.sample(max_size, replace=True)
        df_oversampled = pd.concat([df_oversampled, oversampled_group], axis=0)

    # return oversampled dataframe
    return df_oversampled

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