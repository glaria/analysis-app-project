# Import required libraries
import os
import streamlit as st
import numpy as np
import base64
import plotly.express as px
from app_functions import *

# Set up the Streamlit app
st.set_page_config(page_title="Data Load · Campaign Analysis App", layout="wide", page_icon="📊")
st.title("Data load")

# Create a file uploader for CSV or Excel files
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

# Add an input field for the CSV separator
csv_separator = st.sidebar.text_input("CSV Separator", value=",")

if 'init' not in st.session_state: # a first run init!
    st.session_state.init = True # not related to any widget. Thats why it is preserved!
    st.session_state.store = {} # not related to a widget -> preserved!

# Function to read and process the uploaded file
def process_file(uploaded_file, separator):
    if uploaded_file:
        try:
            if uploaded_file.type == "text/csv":
                data = pd.read_csv(uploaded_file, sep=separator, on_bad_lines='skip')
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Invalid file type. Please upload a CSV or Excel file.")
                return None

            return data
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            return None
    else:
        return None

# Process the uploaded file and convert it into a pandas DataFrame
data = process_file(uploaded_file, csv_separator)

if data is not None and len(data.columns) <= 1:
    st.error("Only one column was detected. Check that the CSV separator is correct.")

if data is not None and len(data.columns) > 1:
    # Reload only when the file (or separator) changes, so in-session edits
    # such as duplicate removal are not overwritten on every rerun
    file_id = (uploaded_file.name, uploaded_file.size, csv_separator)
    if st.session_state.get("uploaded_file_id") != file_id:
        st.session_state.uploaded_file_id = file_id
        st.session_state.uploaded_data = data

    # Preview the session copy, which reflects in-session edits such as duplicate removal
    st.dataframe(st.session_state.uploaded_data.head(5))

    # Infer data types of the loaded dataframe
    inferred_info_dataset = infer_datatypes_and_metatypes(st.session_state.uploaded_data)

    # Editable table for the users to manually define data types and meta types.
    # Keyed by file_id so edits survive reruns but reset when a new file is loaded.
    st.subheader("Define data types and meta types")
    st.caption("Review the inferred role of each column and correct it if needed.")
    user_defined_info_dataset = st.data_editor(
        inferred_info_dataset,
        hide_index=True,
        width="stretch",
        column_config={
            'COLUMN': st.column_config.TextColumn("Column", disabled=True),
            'DATATYPE': st.column_config.SelectboxColumn(
                "Data type", options=['BOOL', 'STRING', 'NUM_ST', 'NUMERIC'], required=True),
            'METATYPE': st.column_config.SelectboxColumn(
                "Meta type", options=['TGCG', 'PK', 'KPI', 'SF'], required=True),
        },
        key=f"info_editor_{uploaded_file.name}_{uploaded_file.size}_{csv_separator}",
    )

    #store info_dataset after user actions
    st.session_state.user_defined_info_dataset = user_defined_info_dataset

    pk_condition = user_defined_info_dataset['METATYPE'] == 'PK'

    if any(pk_condition):
        pk_column = user_defined_info_dataset.loc[pk_condition, 'COLUMN'].values[0]
    else:
        pk_column = None
    

    # Check for duplicate values in the PK column. The removal confirmation is
    # keyed by (file, pk column) in session state so it persists across reruns
    # and resets automatically when a new file or a different PK is chosen.
    if pk_column is not None and pk_column in st.session_state.uploaded_data.columns:
        duplicate_count = int(st.session_state.uploaded_data[pk_column].duplicated().sum())
        dedup_key = (file_id, pk_column)

        if duplicate_count > 0:
            dup_word = "duplicate" if duplicate_count == 1 else "duplicates"
            st.warning(f"Found {duplicate_count:,} {dup_word} in the unique key column ({pk_column}).")
            if st.button("Remove Duplicates"):
                # Remove duplicates while keeping the first record
                st.session_state.uploaded_data = st.session_state.uploaded_data.drop_duplicates(subset=pk_column, keep="first")
                st.session_state.duplicates_removed_for = dedup_key
                # Rerun so the preview and checks above reflect the deduplicated data
                st.rerun()
        elif st.session_state.get("duplicates_removed_for") == dedup_key:
            st.success(f"Duplicates removed. Every value in {pk_column} is unique now.")

    # remove null values based on the data types in `user_defined_info_dataset`
    for index, row in st.session_state.user_defined_info_dataset.iterrows():
        column = row['COLUMN']
        datatype = row['DATATYPE']

        if datatype == 'STRING':
            st.session_state.uploaded_data[column] = st.session_state.uploaded_data[column].fillna('NONE')
        elif datatype == 'NUMERIC' or datatype == 'NUM_ST':
            st.session_state.uploaded_data[column] = st.session_state.uploaded_data[column].fillna(0)

    # Validate the session copy (the frame that will actually be written on Process)
    validation_errors = validate_datatypes_and_metatypes(st.session_state.uploaded_data, user_defined_info_dataset)

    if not validation_errors:
        if st.button("Process Data"):
            st.success("Done, your data is ready.")
            url = 'Analysis_Results'
            st.markdown("Continue to the [Analysis results](%s)" % url)

            st.session_state.store['uploaded_data'] = st.session_state.uploaded_data
            st.session_state.store['user_defined_info_dataset'] = st.session_state.user_defined_info_dataset
            
            os.makedirs("pages/temp", exist_ok=True)  # not tracked by git, may not exist on a fresh clone
            try:
                st.session_state.uploaded_data.to_parquet("pages/temp/uploaded_data.parquet", index=False)
                st.session_state.user_defined_info_dataset.to_parquet("pages/temp/user_defined_info_dataset.parquet", index=False)
            except Exception:
                # parquet needs homogeneous column types; fall back to the legacy CSV handoff
                st.session_state.uploaded_data.to_csv("pages/temp/uploaded_data.csv", sep = ',', mode = 'w+', index = False)
                st.session_state.user_defined_info_dataset.to_csv("pages/temp/user_defined_info_dataset.csv", sep = ',', mode = 'w+', index = False)
    else:
        st.error("Please fix the following before processing:\n" +
                 "\n".join(f"- {error}" for error in validation_errors))

