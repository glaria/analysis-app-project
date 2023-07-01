import streamlit as st
from matplotlib import pyplot as plt
import lightgbm as lgb
#from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from app_functions import *
from sklearn.tree import _tree

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

st.write("We need to define training/test sets before training model")

st.write("It would be nice to do some gridsearch on hyperparameters")


# dataset and information_dataset
dataset = pd.read_csv("pages/temp/uploaded_data.csv", sep = ',')  
information_dataset = pd.read_csv("pages/temp/user_defined_info_dataset.csv", sep = ',') 

# Extract the TGCG column from the dataset
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

# Set to lower TGCG column: TARGET-> target, Control -> control
dataset[tgcg_column] = dataset[tgcg_column].str.lower()

# Create new column with tgcg as flags
dataset['tgcg_fl'] = np.where(dataset[tgcg_column] == 'target', 1, 0)

#we need to iterate over each kpi

# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values

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

dataset_fix = dataset.copy() #store a copy of the dataset

# Conditions before processing
num_records = dataset.shape[0]
num_control_records = dataset[dataset['tgcg_fl'] == 0].shape[0]
num_target_records = dataset[dataset['tgcg_fl'] == 1].shape[0]

if num_records > 200000:
    st.write("The number of total records is greater than 200,000. The model cannot be processed.")
elif num_records < 200:
    st.write("The number of total records is less than 200. The model cannot be processed.")
elif num_control_records < 100:
    st.write("There are fewer than 100 control records. The model cannot be processed.")
elif num_target_records < 100:
    st.write("There are fewer than 100 target records. The model cannot be processed.")
else:
    for kpi in kpi_columns:

        dataset = dataset_fix.copy() #after the first iteration we need to use a clean version of the dataset
        dataset_copy = dataset.copy()
        kpi_original = kpi
        #create model_target column ()
        dataset_copy[kpi + '_model_target'] = dataset_copy.apply(lambda row: 1 if row['tgcg_fl'] == row[kpi] else 0, axis=1)
        kpi = kpi + '_model_target'
        # Create a list with all the columns of interest
        all_columns = list(segmentation_columns) + list(continue_segmentation_columns) + ['tgcg_fl'] + [kpi]

        # Create a new DataFrame that only contains the columns of interest
        df_subset = dataset_copy[all_columns]

        # Apply the oversample function to the new DataFrame
        df_oversampled = oversample(df_subset, [kpi, 'tgcg_fl'])

        # Get counts of each group
        #counts = df_oversampled.groupby('tgcg_fl')[kpi].value_counts()
        #st.write(counts)

        # Get percentages of each group
        #percentages = df_oversampled.groupby('tgcg_fl')[kpi].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
        #st.write(percentages)

        #train model
        # Definir las características (X) y la variable objetivo (y)
        X = df_oversampled.drop(columns=['tgcg_fl', kpi])
        y = df_oversampled[kpi]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # train model
        model = LGBMRegressor(max_depth = 5, n_estimators = 100,  min_child_samples =50) #ponemos un número relativamente alto de min_child_samples porque al sobremuestrear nuestro dataset crece mucho
        model.fit(X, y)

        # Hacer predicciones en el conjunto de prueba
        #y_pred = model.predict(X_test)

        # Create a new list of columns that only includes columns present in 'dataset'
        dataset_columns = list(segmentation_columns) + list(continue_segmentation_columns)

        # Define the features (X) for the original dataset
        X_dataset = dataset[dataset_columns]

        # Make predictions on the original dataset
        dataset_predictions = model.predict(X_dataset)

        # You can add these predictions as a new column in your original DataFrame
        dataset['predictions'] = dataset_predictions

        # Remove "_model_target" from the kpi variable and set the title of the plot
        kpi_name = kpi.replace("_model_target", "")
        st.header(f"Model scores of the kpi: {kpi_name}")
        st.write(dataset.drop(columns = ['tgcg_fl']))


        # After creating the predictions column in the original dataset
        # Define the 25th and 75th percentiles
        upper_quartile = np.percentile(dataset['predictions'], 75)
        lower_quartile = np.percentile(dataset['predictions'], 25)

        # Create a new binary column where 'Top25%' indicates the prediction is above the upper quartile
        # and 'Bottom25%' indicates the prediction is below the lower quartile
        dataset['top_bottom'] = np.where(dataset['predictions'] >= upper_quartile, 'Top25%', np.where(dataset['predictions'] <= lower_quartile, 'Bottom25%', np.nan))

        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])

        # Loop to calculate the metrics for each group (Top25% and Bottom25%)
        for value in ['Top25%','Bottom25%']:
            filtered_dataset = dataset.loc[dataset['top_bottom'] == value]
            result_df = calculate_metrics2(filtered_dataset, kpi_original, tgcg_column)
            if not result_df.empty:
                for index, row in result_df.iterrows():
                    new_row = pd.DataFrame({
                        "KPI": [kpi_original],
                        "TG Acceptors": [row["TG Acceptors"]],
                        "TG Acceptance (%)": [row["TG Acceptance (%)"]],
                        "CG Acceptors": [row["CG Acceptors"]],
                        "CG Acceptance (%)": [row["CG Acceptance (%)"]],
                        "Uplift (%)": [row["Uplift (%)"]],
                        "P-value": [row["P-value"]],
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

            results_df["TG Acceptors"] = results_df["TG Acceptors"].astype(float).round(0).astype(int)
            results_df["CG Acceptors"] = results_df["CG Acceptors"].astype(float).round(0).astype(int)

        st.markdown(f"**Results model**")
        st.dataframe(results_df.style.apply(highlight_pvalue, axis=1))


        #download link
        filename = f"model_kpi_{kpi_name}.csv"
        st.markdown(download_csv_link(dataset.drop(columns = ['tgcg_fl','top_bottom']), filename), unsafe_allow_html=True)


        ### *** Explanining the lgbm using a simple decision tree *** ###

        fig, ax = plt.subplots(figsize=(12, 10))  # Create a subplot to adjust the size if necessary
        lgb.plot_importance(model, ax=ax)  # Pass the axis to the function to plot

        ax.set_title(f"Feature importance of the kpi {kpi_name}")
        st.pyplot(fig)  # Display the figure in Streamlit