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

# Streamlit theme configuration
st.set_page_config(
    page_title="FOB Prediction",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Centered logo using st.image
col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns to center the image

with col2:  # Center column
    st.image("IND Logo PNG + (1).png", width=300)  # Set the width to a smaller size

# Set title
st.title('FOB Prediction')

# Show cleaned data in the app
st.subheader('FOB Data')
st.write(cleaned_data)

# Create input fields for user to input data
style = st.selectbox('Select Style', cleaned_data['STYLE'].unique())
department = st.selectbox('Select Department', cleaned_data['Department'].unique())
product_des = st.selectbox('Product Description', cleaned_data['PRODUCT DES.'].unique())
buyer = st.selectbox('Select Buyer', cleaned_data['BUYER'].unique())
country = st.selectbox('Select Country', cleaned_data['CONTRY'].unique())

# Input field for ORDER QTY as a number
order_qty = st.number_input('Enter Order Quantity', min_value=1, value=1, step=1)

# Prediction button
if st.button('Predict Your FOB'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'STYLE': [style],
        'Department': [department],
        'PRODUCT DES.': [product_des],
        'ORDER QTY': [order_qty],
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
        best_model = None
        min_relative_error = float('inf')
        exact_match_found = False
        actual_fob = None  # Initialize actual_fob to store the matched FOB

        for model_name, prediction in predictions.items():
            # Match the input data with the cleaned data for actual FOB
            matches = cleaned_data[
                (cleaned_data['STYLE'] == style) &
                (cleaned_data['Department'] == department) &
                (cleaned_data['PRODUCT DES.'].str.contains(product_des, case=False)) &
                (cleaned_data['ORDER QTY'] == order_qty) &
                (cleaned_data['BUYER'] == buyer) &
                (cleaned_data['CONTRY'] == country)
            ]

            # Check for exact match
            if not matches.empty:
                exact_match_found = True
                actual_fob = matches['FOB'].values[0]  # Assuming you want the FOB of the first match
                relative_error = calculate_relative_error(actual_fob, prediction)

                # Check for the model with the least relative error
                if relative_error < min_relative_error:
                    min_relative_error = relative_error
                    best_model = model_name

            # Display predictions
            st.markdown(f'<div style="background-color: #F7F7F7; padding: 1rem; border-radius: 8px;">{model_name} Prediction: {prediction}</div>', unsafe_allow_html=True)

        # Display actual FOB once
        if exact_match_found and actual_fob is not None:
            st.markdown(f'<div style="color: blue; font-weight: bold; font-size: 1.2rem; text-align: center; margin: 1rem 0;">Actual FOB: {actual_fob}</div>', unsafe_allow_html=True)

        # Display best model highlight
        if exact_match_found:
            st.markdown(f'<div style="color: #32cd32; font-weight: bold; text-align: center;">Exact Match Found!</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #FF8C02; font-weight: bold;">{best_model} has the least relative error: {min_relative_error:.2f}%!</div>', unsafe_allow_html=True)
        else:
            st.write("No exact matches found for the predictions. Here are the predicted values:")

    except ValueError as e:
        st.error(f"Error during prediction: {e}")
