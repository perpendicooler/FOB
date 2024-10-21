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

# CSS styling for light color scheme, center alignment, animations, and box designs
st.markdown("""
    <style>
    body {
        background-color: #F5F5F5;  /* Light Grey Background */
        font-family: Arial, sans-serif;
        color: #2F4F4F; /* Dark Slate Gray for text */
    }
    .center-title {
        text-align: center;
        animation: fadeIn 2s ease-in-out;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButton button {
        background-color: #6A99D3; /* Soft Slate Blue */
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
        transition: background-color 0.3s, transform 0.3s;
    }
    .stButton button:hover {
        background-color: #4A7D9D; /* Darker Slate Blue on hover */
        transform: scale(1.05);
    }
    .prediction-box {
        background-color: white;  /* Soft White for boxes */
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 20px;
        font-weight: bold;
        color: #2F4F4F; /* Text color */
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease-in-out;
    }
    .exact-match {
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes glow {
        from {
            box-shadow: 0 0 10px #FFD700;
        }
        to {
            box-shadow: 0 0 20px #FFD700, 0 0 30px #FFD700, 0 0 40px #FFD700;
        }
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

        # Display predictions in elegant boxes
        st.subheader('Predictions')
        st.markdown(f'<div class="prediction-box">Linear Regression Prediction: {predictions["Linear Regression"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-box">Random Forest Prediction: {predictions["Random Forest"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-box">Gradient Boosting Prediction: {predictions["Gradient Boosting"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-box">XGBoost Prediction: {predictions["XGBoost"]}</div>', unsafe_allow_html=True)

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
            st.markdown('<div class="center-title exact-match">Exact Match Found!</div>', unsafe_allow_html=True)
            st.write(f"Actual FOB: {actual_fob}")

            for model_name, prediction in predictions.items():
                relative_error = calculate_relative_error(actual_fob, prediction)
                st.write(f'Relative Error for {model_name}: {relative_error:.2f}%')
        else:
            st.write("No exact matches found for the predictions.")

    except ValueError as e:
        st.error(f"Error during prediction: {e}")
st.markdown('</div>', unsafe_allow_html=True)
