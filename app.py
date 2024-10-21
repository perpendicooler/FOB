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
        /* Theme Settings */
        .app-container {
            background-color: #00325B; /* Dark Blue */
            padding: 2rem;
            border-radius: 10px;
            color: white; /* White text color for readability */
        }
        .outer-box {
            background-color: #e0e0e0; /* Very light grey for the outer box */
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
        }
        h1 {
            text-align: center;
            font-size: 2rem;  /* Shorter font size for the title */
            color: #FF8C02; /* Bright Orange */
        }
        .prediction-box {
            background-color: #ffffff; /* White for boxes */
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
            color: #FF8C02; /* Highlight color for best prediction */
            font-weight: bold;
        }
        .exact-match {
            color: #32cd32; /* Light green for exact match */
            font-weight: bold;
            text-align: center;
            animation: glow 1s infinite;
            margin: 1rem 0;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        @keyframes glow {
            0% { text-shadow: 0 0 5px #32cd32, 0 0 10px #32cd32; }
            50% { text-shadow: 0 0 20px #32cd32, 0 0 30px #32cd32; }
            100% { text-shadow: 0 0 5px #32cd32, 0 0 10px #32cd32; }
        }
        .actual-fob {
            color: blue; /* Different font color for actual FOB */
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered logo using st.image
col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns to center the image

with col2:  # Center column
    st.image("IND Logo PNG + (1).png", width=300)  # Set the width to a smaller size

# Set title (shorter than the logo)
st.title('FOB Prediction')

# Show cleaned data in the app
st.subheader('FOB Data')
st.write(cleaned_data)

# Create an outer box for input fields and button
with st.container():
    st.markdown('<div class="outer-box">', unsafe_allow_html=True)

    # Input fields for user to input data
    style = st.selectbox('Select Style', cleaned_data['STYLE'].unique())
    department = st.selectbox('Select Department', cleaned_data['Department'].unique())
    product_des = st.selectbox('Product Description', cleaned_data['PRODUCT DES.'].unique())
    buyer = st.selectbox('Select Buyer', cleaned_data['BUYER'].unique())
    country = st.selectbox('Select Country', cleaned_data['CONTRY'].unique())

    # Input field for ORDER QTY as a number
    order_qty = st.number_input('Enter Order Quantity', min_value=1, value=1, step=1)

    # Center the button using a div
    st.markdown('<div class="centered">', unsafe_allow_html=True)

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

                # Display predictions in boxes
                st.markdown(f'<div class="prediction-box"> {model_name} Prediction: {prediction} </div>', unsafe_allow_html=True)

            # Display actual FOB once
            if exact_match_found and actual_fob is not None:
                st.markdown(f'<div class="actual-fob">Actual FOB: {actual_fob}</div>', unsafe_allow_html=True)

            # Display best model highlight
            if exact_match_found:
                st.markdown(f'<div class="exact-match">Exact Match Found!</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="highlight">{best_model} has the least relative error: {min_relative_error:.2f}%!</div>', unsafe_allow_html=True)
            else:
                st.write("No exact matches found for the predictions. Here are the predicted values:")

        except ValueError as e:
            st.error(f"Error during prediction: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close centered div for button
    st.markdown('</div>', unsafe_allow_html=True)  # Close outer box div
