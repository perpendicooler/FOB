import streamlit as st
import pandas as pd
import joblib

# Load the models
lr_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Load the cleaned data
file_path = 'cleaned_dataset.xlsx'
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

# CSS styling for center alignment and animation
st.markdown("""
    <style>
    .center-title {
        text-align: center;
        font-family: Arial, sans-serif;
        color: #4B0082;
        animation: fadeIn 2s ease-in-out;
    }
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButton button {
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton button:hover {
        background-color: #4B0082;
        color: white;
        transform: scale(1.05);
    }
    .stSelectbox select {
        transition: all 0.3s ease;
    }
    .stSelectbox:hover select {
        background-color: #f0f0f0;
        transform: scale(1.02);
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Center the title
st.markdown('<h1 class="center-title">FOB Prediction App</h1>', unsafe_allow_html=True)

# Show cleaned data in the app
st.subheader('FOB Data')
st.write(cleaned_data)

# Input fields for user to input data
style = st.selectbox('Select Style', cleaned_data['STYLE'].unique())
department = st.selectbox('Select Department', cleaned_data['Department'].unique())
product_des = st.selectbox('Product Description', cleaned_data['PRODUCT DES.'].unique())
buyer = st.selectbox('Select Buyer', cleaned_data['BUYER'].unique())
country = st.selectbox('Select Country', cleaned_data['CONTRY'].unique())

# Input field for ORDER QTY as a number
order_qty = st.number_input('Enter Order Quantity', min_value=1, value=1, step=1)

# Center align the prediction button
st.markdown('<div class="center-button">', unsafe_allow_html=True)
if st.button('Predict FOB'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'STYLE': [style],
        'Department': [department],
        'PRODUCT DES.': [product_des],
        'ORDER QTY': [order_qty],  # Use the order quantity from the number input
        'BUYER': [buyer],
        'CONTRY': [country]
    })

    # Strip spaces from the input data column names
    input_data.columns = input_data.columns.str.strip()

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
            (cleaned_data['ORDER QTY'] == order_qty) &  # Match the order quantity
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
st.markdown('</div>', unsafe_allow_html=True)
