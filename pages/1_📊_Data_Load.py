# Import required libraries
import streamlit as st
import numpy as np
import base64
import plotly.express as px
from app_functions import *

# Set up the Streamlit app
st.set_page_config(page_title="CRM Campaign Analysis App", layout="wide", page_icon= "📊")
st.title("CRM Campaign Analysis App")

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
                data = pd.read_csv(uploaded_file, sep=separator, error_bad_lines=False)
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
try:
    if "uploaded_data" not in st.session_state and len(data.columns) > 1:
        st.session_state.uploaded_data = data
except Exception:
    pass
try: #show the dataframe but if no file loaded we hide the error
    st.dataframe(data.head(5))
except Exception:
    pass
# Infer data types of the loaded dataframe
if data is not None and len(data.columns) > 1:

    inferred_info_dataset = infer_datatypes_and_metatypes(data)

    # Create an empty container to display the data types and meta types
    data_types_metatypes_container = st.empty()
    data_types_metatypes_container.write("Data types and meta types:")
    # Function to display the data types and meta types in the container
    def display_data_types_metatypes(info_dataset):

        data_types_metatypes_container.write(info_dataset)

    # Display the inferred data types and meta types
    display_data_types_metatypes(inferred_info_dataset)

    # Create dropdown menus for users to manually define data types and meta types
    st.sidebar.header("Define data types and meta types")
    user_defined_info_data = {'COLUMN': [], 'DATATYPE': [], 'METATYPE': []}

    for index, row in inferred_info_dataset.iterrows():
        column = row['COLUMN']
        inferred_datatype = row['DATATYPE']
        inferred_metatype = row['METATYPE']

        # Create two columns to display the dropdown menus side by side
        col1, col2 = st.sidebar.columns(2)

        user_datatype = col1.selectbox(
            f"Data Type for {column}",
            options=['BOOL', 'STRING', 'NUM_ST', 'NUMERIC'],
            index=['BOOL', 'STRING', 'NUM_ST', 'NUMERIC'].index(inferred_datatype),
            key=f"{column}_datatype"
        )

        user_metatype = col2.selectbox(
            f"Meta Type for {column}",
            options=['TGCG', 'PK', 'KPI', 'SF'],
            index=['TGCG', 'PK', 'KPI', 'SF'].index(inferred_metatype),
            key=f"{column}_metatype"
        )

        user_defined_info_data['COLUMN'].append(column)
        user_defined_info_data['DATATYPE'].append(user_datatype)
        user_defined_info_data['METATYPE'].append(user_metatype)

    # Update the displayed data types and meta types when the user makes changes
    user_defined_info_dataset = pd.DataFrame(user_defined_info_data)

    #store info_dataset after user actions
    st.session_state.user_defined_info_dataset = user_defined_info_dataset

    display_data_types_metatypes(user_defined_info_dataset)
    pk_condition = user_defined_info_dataset['METATYPE'] == 'PK'

    if any(pk_condition):
        pk_column = inferred_info_dataset.loc[pk_condition, 'COLUMN'].values[0]
    else:
        pk_column = None
    

# Check for duplicate values in the PK column and display a warning message in red if duplicates are found
    warning_message_container = st.empty()

    if pk_column is not None and pk_column in st.session_state.uploaded_data.columns:

        if st.session_state.uploaded_data[pk_column].duplicated().any() and pk_column is not None:
            warning_message_container.markdown(f'<span style="color:red">Warning: Duplicate values found in the unique key column ({pk_column})!!</span>', unsafe_allow_html=True)

        # Create a button to remove duplicates
            if st.button("Remove Duplicates"):
                # Remove duplicates while keeping the first record
                st.session_state.uploaded_data = st.session_state.uploaded_data.drop_duplicates(subset=pk_column, keep="first")

            #st.success("Duplicates removed successfully!")

            # Update the warning message
                if not st.session_state.uploaded_data[pk_column].duplicated().any():
                    warning_message_container.markdown(f'<span style="color:green">No duplicate values found in the unique key column ({pk_column}).</span>', unsafe_allow_html=True)
                else:
                    warning_message_container.markdown(f'<span style="color:red">Warning: Duplicate values found in the unique key column ({pk_column})!!</span>', unsafe_allow_html=True)

    # Validate data types and meta types
    is_valid = validate_datatypes_and_metatypes(data, user_defined_info_dataset)

    # remove null values based on the data types in `user_defined_info_dataset` 
    for index, row in st.session_state.user_defined_info_dataset.iterrows():
        column = row['COLUMN']
        datatype = row['DATATYPE']

        if datatype == 'STRING':
            st.session_state.uploaded_data[column] = st.session_state.uploaded_data[column].fillna('NONE')
        elif datatype == 'NUMERIC' or datatype == 'NUM_ST':
            st.session_state.uploaded_data[column] = st.session_state.uploaded_data[column].fillna(0)

    if is_valid:
        if st.button("Process Data"):
            st.success("Data processing started!")
            url = 'Analysis_Results'
            #st.write("check out this [link](%s)" % url)
            st.markdown("Go to the [Analysis results](%s)" % url)

            st.session_state.store['uploaded_data'] = st.session_state.uploaded_data
            st.session_state.store['user_defined_info_dataset'] = st.session_state.user_defined_info_dataset
            
            st.session_state.uploaded_data.to_csv("pages/temp/uploaded_data.csv", sep = ',', mode = 'w+')
            st.session_state.user_defined_info_dataset.to_csv("pages/temp/user_defined_info_dataset.csv", sep = ',', mode = 'w+')
    else:
        st.error("There is an issue with the data types or meta types. Please review and correct them.")

