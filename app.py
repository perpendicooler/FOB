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

# CSS styling for light color scheme
st.markdown(
    """
    <style>
        .app-container {
            background-color: #f5f5f5; /* Light grey background */
            padding: 2rem;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            color: #333;
        }
        .prediction-box {
            background-color: #e0e0e0; /* Light grey for boxes */
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .prediction-box:hover {
            transform: scale(1.02);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
        }
        .highlight {
            color: green; /* Highlight color for best prediction */
            font-weight: bold;
        }
        .exact-match {
            color: #32cd32; /* Light green for exact match */
            font-weight: bold;
            text-align: center;
            animation: glow 1s infinite;
            margin: 1rem 0;
        }
        @keyframes glow {
            0% { text-shadow: 0 0 5px #32cd32, 0 0 10px #32cd32; }
            50% { text-shadow: 0 0 20px #32cd32, 0 0 30px #32cd32; }
            100% { text-shadow: 0 0 5px #32cd32, 0 0 10px #32cd32; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

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

# Input field for ORDER QTY as a number
order_qty = st.number_input('Enter Order Quantity', min_value=1, value=1, step=1)

# Prediction button
if st.button('Predict FOB'):
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

            # Calculate and display relative errors
            if not matches.empty:
                exact_match_found = True
                actual_fob = matches['FOB'].values[0]  # Assuming you want the FOB of the first match
                relative_error = calculate_relative_error(actual_fob, prediction)

                # Check for the model with the least relative error
                if relative_error < min_relative_error:
                    min_relative_error = relative_error
                    best_model = model_name

            # Display predictions in boxes
            st.markdown(f'<div class="prediction-box"> {model_name} Prediction: {prediction} </div>', unsafe_allow_html=True)

        # Display best model highlight
        if best_model:
            st.markdown(f'<div class="exact-match">Exact Match Found!</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="highlight">{best_model} has the least relative error!</div>', unsafe_allow_html=True)
        elif not exact_match_found:
            st.write("No exact matches found for the predictions. Here are the predicted values:")
        else:
            st.write("No exact matches found for the predictions.")

    except ValueError as e:
        st.error(f"Error during prediction: {e}")
