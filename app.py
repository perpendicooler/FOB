import streamlit as st
import pandas as pd
import joblib

# Load the models
lr_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Load the cleaned data
file_path = 'FOB_cleaned.xlsx'
cleaned_data = pd.read_excel(file_path)

# Strip spaces from column names
cleaned_data.columns = cleaned_data.columns.str.strip()

# Function to clean data
def clean_data(data):
    data['BUYER'] = data['BUYER'].str.upper()  # Standardize buyer names
    data = data[(data != 0).all(axis=1)]  # Drop rows with any 0 values
    return data

# Function to calculate relative error
def calculate_relative_error(actual, predicted):
    return abs((actual - predicted) / actual) * 100  # Relative error in percentage

# Clean the data
cleaned_data = clean_data(cleaned_data)

# Set title
st.title('FOB Prediction App')

# Show cleaned data in the app
st.subheader('FOB Data')
st.write(cleaned_data)

# Input fields for user to input data
style = st.selectbox('Select Style', cleaned_data['STYLE'].unique())
department = st.selectbox('Select Department', cleaned_data['Department'].unique())
product_des = st.selectbox('Product Description', cleaned_data['PRODUCT DES.'].unique())
buyer = st.selectbox('Select Buyer', cleaned_data['BUYER'].unique())
country = st.selectbox('Select Country', cleaned_data['CONTRY'].unique())

# Filter the data for ORDER QTY based on the user's selections
filtered_data = cleaned_data[
    (cleaned_data['STYLE'] == style) &
    (cleaned_data['Department'] == department) &
    (cleaned_data['PRODUCT DES.'].str.contains(product_des, case=False)) &
    (cleaned_data['BUYER'] == buyer) &
    (cleaned_data['CONTRY'] == country)
]

# Selectbox for ORDER QTY
if not filtered_data.empty:
    order_qty = st.selectbox('Select Order Quantity', filtered_data['ORDER QTY'].unique())
else:
    st.write("No available Order Quantities for the selected filters.")

# Prediction button
if st.button('Predict FOB'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'STYLE': [style],
        'Department': [department],
        'PRODUCT DES.': [product_des],
        'ORDER QTY': [order_qty],  # Use the order quantity from the selectbox
        'BUYER': [buyer],
        'CONTRY': [country]
    })

    # Strip spaces from the input data column names
    input_data.columns = input_data.columns.str.strip()

    # Print column names for debugging
    st.write("Cleaned Data Columns:", cleaned_data.columns.tolist())
    st.write("Input Data Columns:", input_data.columns.tolist())

    # Make predictions using the models
    try:
        predictions = {
            'Linear Regression': lr_model.predict(input_data)[0],
            'Random Forest': rf_model.predict(input_data)[0],
            'Gradient Boosting': gb_model.predict(input_data)[0],
            'XGBoost': xgb_model.predict(input_data)[0]
        }

        # Display predictions
        st.subheader('Predictions')
        for model_name, prediction in predictions.items():
            st.write(f'{model_name} Prediction: {prediction}')

        # Match the input data with the cleaned data for actual FOB
        matches = cleaned_data[
            (cleaned_data['STYLE'] == style) &
            (cleaned_data['Department'] == department) &
            (cleaned_data['PRODUCT DES.'].str.contains(product_des, case=False)) &
            (cleaned_data['ORDER QTY'] == order_qty) &
            (cleaned_data['BUYER'] == buyer) &
            (cleaned_data['CONTRY'] == country)
        ]

        # Calculate and display relative errors if matches are found
        if not matches.empty:
            actual_fob = matches['FOB'].values[0]  # Assuming you want the FOB of the first match
            for model_name, prediction in predictions.items():
                relative_error = calculate_relative_error(actual_fob, prediction)
                st.write(f'Relative Error for {model_name}: {relative_error:.2f}%')
        else:
            st.write("No exact matches found for the predictions.")

    except ValueError as e:
        st.error(f"Error during prediction: {e}")
