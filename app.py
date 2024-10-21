import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Add custom CSS for styling with enhancements and animations
# Inject custom styles
st.markdown(
    """
    <style>
    /* Set the background color for the entire page to a very light grey */
    section.main {
        background-color: #F9F9F9; /* Very light grey */
        color: #333333;
        padding: 1px;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    
    .stApp {
        background-color: #F9F9F9;  /* Very light grey */
        font-family: Arial, sans-serif; /* Set font to Arial */
    }

    /* Style for title */
    .title {
        font-size: 1.5em; /* Smaller title size */
        color: #2C3E50; /* Darker blue color for professionalism */
        font-weight: bold;
        text-align: center;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }

    # /* Style for buttons */
    # .stButton {
    #     #background-color: #4CAF50; /* Dark green */
    #     color: black;
    #     padding: 12px 24px;
    #     border: none;
    #     border-radius: 8px;
    #     font-size: 1.1em; /* Slightly smaller font size */
    #     box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    #     transition: background-color 0.3s, transform 0.2s, box-shadow 0.2s ease;
    #     font-family: Arial, sans-serif; /* Set font to Arial */
    #     cursor: pointer; /* Change cursor to pointer */
    #     display: block; /* Block display to allow centering */
    #     margin: 0 auto; /* Center the button */
    # }
    
    # .stButton:hover {
    #     background-color: #45a049; /* Slightly lighter green on hover */
    #     transform: translateY(-3px);
    #     box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    # }
    
    /* Style for input fields */
    .input-field {
        margin: 10px 0;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        width: 100%;
        max-width: 400px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    
    /* Style for metrics box */
    .metrics {
        margin: 20px 0;
        padding: 15px;
        border-radius: 5px;
        background-color: #BEBEBE; /* Grey background */
        color: #333333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease;
        font-family: Arial, sans-serif; /* Set font to Arial */
    }
    .metrics:hover {
        background-color: #DC143C; /* Darker grey on hover */
    }
    
    /* Center align the tabs */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center; /* Center align the tabs */
        gap: 30px; /* Adjust the spacing between tabs */
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Load the trained models
model_xgboost = joblib.load('trained_xgboost_model.pkl')
model_rf = joblib.load('trained_random_forest_model.pkl')

# Load the dataset for reference
file_path = 'SMV.xlsx'
data = pd.read_excel(file_path)

# Define categorical and numerical features
categorical_features = ['GG', 'Operation', 'Operation Position', 'Operation Description',
                        'Fiber 1', 'Fiber 2', 'Fiber 3', 'Knit Construction',
                        'Count 1', 'Count 2', 'Count 3']
numerical_features = ['Percentage 1', 'Percentage 2', 'Percentage 3',
                      'Ply 1', 'Ply 2', 'Ply 3', 'MC Speed', 'Length (cm)']

# Create tabs for different sections of the app
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠SMV Prediction", "🚀Overview", 
                                        "📊Data Preparation", "💻Modeling", "📈Results"])

with tab1:
    # Centered logo using st.image
    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns to center the image

    with col2:  # Center column
        st.image("IND Logo PNG +.png", width=300)  # Set the width to a smaller size

    st.markdown('<h1 class="title">SMV Prediction</h1>', unsafe_allow_html=True)
    # Dynamic selection of number of fibers
    num_fibers = st.selectbox('Select Number of Fibers', [1, 2, 3])
    # Input fields for predictions
    GG = st.radio('Select GG', data['GG'].unique().tolist())
    Operation = st.selectbox('Select Operation', data['Operation'].unique().tolist())
    Operation_Position = st.selectbox('Select Operation Position', data['Operation Position'].unique().tolist())
    Operation_Description = st.selectbox('Select Operation Description', data['Operation Description'].unique().tolist())
    Knit_Construction = st.selectbox('Type Knit Construction', data['Knit Construction'].unique().tolist())



    # Initialize variables for all fiber inputs
    Percentage_1, Fiber_1, Count_1, Ply_1 = None, None, None, None
    Percentage_2, Fiber_2, Count_2, Ply_2 = 0, None, 0, 0  # Default values
    Percentage_3, Fiber_3, Count_3, Ply_3 = 0, None, 0, 0  # Default values

    # Fiber 1 inputs
    if num_fibers >= 1:
        Percentage_1 = st.number_input('Enter Percentage 1', min_value=0.0, max_value=100.0, step=0.1)
        Fiber_1 = st.selectbox('Select Fiber 1', data['Fiber 1'].unique().tolist())
        Count_1 = st.selectbox('Select Count 1', data['Count 1'].unique().tolist())
        Ply_1 = st.number_input('Enter Ply 1', min_value=0)

    # Fiber 2 inputs
    if num_fibers >= 2:
        Percentage_2 = st.number_input('Enter Percentage 2', min_value=0.0, max_value=100.0, step=0.1)
        Fiber_2 = st.selectbox('Select Fiber 2', data['Fiber 2'].unique().tolist())
        Count_2 = st.selectbox('Select Count 2', data['Count 2'].unique().tolist())
        Ply_2 = st.number_input('Enter Ply 2', min_value=0)

    # Fiber 3 inputs
    if num_fibers == 3:
        Percentage_3 = st.number_input('Enter Percentage 3', min_value=0.0, max_value=100.0, step=0.1)
        Fiber_3 = st.selectbox('Select Fiber 3', data['Fiber 3'].unique().tolist())
        Count_3 = st.selectbox('Select Count 3', data['Count 3'].unique().tolist())
        Ply_3 = st.number_input('Enter Ply 3', min_value=0)

    MC_Speed = st.selectbox('Select MC Speed', data['MC Speed'].unique().tolist())
    Length = st.number_input('Enter Length (cm)', min_value=0.0, max_value=3000.0, step=1.0)
    left, middle, right = st.columns(3)

    # Check if any required inputs are missing
    if middle.button('Predict SMV'):
        missing_values = []
        if not GG:
            missing_values.append("GG")
        if not Operation:
            missing_values.append("Operation")
        if not Operation_Position:
            missing_values.append("Operation Position")
        if not Operation_Description:
            missing_values.append("Operation Description")
        if not Knit_Construction:
            missing_values.append("Knit Construction")
        if num_fibers >= 1 and (Percentage_1 is None or Fiber_1 is None or Count_1 is None or Ply_1 is None):
            missing_values.append("Fiber 1 details")
        if num_fibers >= 2 and (Percentage_2 is None or Fiber_2 is None or Count_2 is None or Ply_2 is None):
            missing_values.append("Fiber 2 details")
        if num_fibers == 3 and (Percentage_3 is None or Fiber_3 is None or Count_3 is None or Ply_3 is None):
            missing_values.append("Fiber 3 details")
        if MC_Speed is None:
            missing_values.append("MC Speed")
        if Length is None:
            missing_values.append("Length")

        # Display warning if any values are missing
        if missing_values:
            st.warning(f"Please fill in the following fields before predicting: {', '.join(missing_values)}")
        else:
            # Prepare the input data
            input_data = pd.DataFrame({
                'GG': [GG], 'Operation': [Operation], 'Operation Position': [Operation_Position],
                'Operation Description': [Operation_Description], 'Knit Construction': [Knit_Construction],
                'Percentage 1': [Percentage_1], 'Fiber 1': [Fiber_1], 'Count 1': [Count_1], 'Ply 1': [Ply_1],
                'Percentage 2': [Percentage_2], 'Fiber 2': [Fiber_2], 'Count 2': [Count_2], 'Ply 2': [Ply_2],
                'Percentage 3': [Percentage_3], 'Fiber 3': [Fiber_3], 'Count 3': [Count_3], 'Ply 3': [Ply_3],
                'MC Speed': [MC_Speed], 'Length (cm)': [Length]
            })

            # One-hot encode the input data
            input_encoded = pd.get_dummies(input_data, columns=data.select_dtypes(include=['object']).columns)

            # Ensure that all model columns are present
            model_columns = pd.get_dummies(data.drop(columns=['SMV'])).columns
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            # Convert to numpy array for prediction
            input_encoded_np = input_encoded.values.astype(np.float32)

            # Model predictions
            # Model predictions
            with st.spinner('Processing your prediction...'):
                try:
                    # Random Forest prediction
                    prediction_rf = model_rf.predict(input_encoded_np)[0]
                    # XGBoost prediction
                    prediction_xgboost = model_xgboost.predict(input_encoded_np)[0]
            
                    st.write(f"**Random Forest Predicted SMV:** {prediction_rf:.2f}")
                    st.write(f"**XGBoost Predicted SMV:** {prediction_xgboost:.2f}")
            
                    # Average the predictions
                    combined_prediction = (prediction_rf + prediction_xgboost) / 2
                    st.write(f"**On average, the SMV is estimated to be around** {combined_prediction:.2f}")
            
                    # Search for actual SMV in the existing data
                    matching_row = data[
                        (data['GG'] == GG) &
                        (data['Operation'] == Operation) &
                        (data['Operation Position'] == Operation_Position) &
                        (data['Operation Description'] == Operation_Description) &
                        (data['Knit Construction'] == Knit_Construction) &
                        (data['MC Speed'] == MC_Speed) &
                        (data['Length (cm)'] == Length) &
                        (data['Percentage 1'] == Percentage_1) &
                        (data['Fiber 1'] == Fiber_1) &
                        (data['Count 1'] == Count_1) &
                        (data['Ply 1'] == Ply_1) &
                        (data['Percentage 2'] == Percentage_2) &
                        (data['Fiber 2'] == Fiber_2) &
                        (data['Count 2'] == Count_2) &
                        (data['Ply 2'] == Ply_2) &
                        (data['Percentage 3'] == Percentage_3) &
                        (data['Fiber 3'] == Fiber_3) &
                        (data['Count 3'] == Count_3) &
                        (data['Ply 3'] == Ply_3)
                    ]
            
                    if not matching_row.empty:
                        actual_smv = matching_row['SMV'].values[0]
                        st.write(f"**Exact match found! Actual SMV:** {actual_smv:.2f}")
            
                        # Calculate errors for both models
                        error_rf = abs(prediction_rf - actual_smv)
                        error_xgboost = abs(prediction_xgboost - actual_smv)
            
                        # Calculate relative errors
                        relative_error_rf = (error_rf / actual_smv) * 100 if actual_smv != 0 else 0
                        relative_error_xgboost = (error_xgboost / actual_smv) * 100 if actual_smv != 0 else 0
            
                        # Display errors using styled metrics
                        st.markdown(f"<div class='metrics'><strong>Random Forest:</strong><br>Point Difference: {error_rf:.2f}<br>Relative Error: {relative_error_rf:.2f}%</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metrics'><strong>XGBoost:</strong><br>Point Difference: {error_xgboost:.2f}<br>Relative Error: {relative_error_xgboost:.2f}%</div>", unsafe_allow_html=True)
            
                        # Determine which model performed better
                        if error_rf < error_xgboost:
                            st.success("**Random Forest** is the better fit for this prediction.")
                        else:
                            st.success("**XGBoost** is the better fit for this prediction.")
            
                    else:
                        st.write("**New combination detected!** No actual SMV available.")
                      #  st.write(f"**On average, the SMV is estimated to be around** {combined_prediction:.2f}")
            
                #     # Save prediction to Excel
                #     if middle.button("Save Prediction"):
                        
                #         predictions_df = pd.DataFrame({
                #             'GG': [GG],
                #             'Operation': [Operation],
                #             'Operation Position': [Operation_Position],
                #             'Operation Description': [Operation_Description],
                #             'Knit Construction': [Knit_Construction],
                #             'Percentage 1': [Percentage_1],
                #             'Fiber 1': [Fiber_1],
                #             'Count 1': [Count_1],
                #             'Ply 1': [Ply_1],
                #             'Percentage 2': [Percentage_2],
                #             'Fiber 2': [Fiber_2],
                #             'Count 2': [Count_2],
                #             'Ply 2': [Ply_2],
                #             'Percentage 3': [Percentage_3],
                #             'Fiber 3': [Fiber_3],
                #             'Count 3': [Count_3],
                #             'Ply 3': [Ply_3],
                #             'MC Speed': [MC_Speed],
                #             'Length (cm)': [Length],
                #             'RF_Predicted_SMV': [prediction_rf],
                #             'XGBoost_Predicted_SMV': [prediction_xgboost]
                #         })
            
                #         predictions_df.to_excel('Prediction_History.xlsx', index=False)
                #         st.success("Prediction saved successfully!")
            
                # except Exception as e:
                #     st.error(f"An error occurred: {str(e)}")


with tab2:
    #st.markdown("## Overview of the SMV Prediction Project")
    st.header("🚀 The Journey of Predicting SMV: From Data to Results")
    
    st.write("""
        **What is SMV?**  
        Standard Minute Value (SMV) is a critical measure in the textile industry that determines the time required to complete a specific operation. By quantifying SMV, industries can better estimate labor costs, optimize operations, and improve overall productivity. It’s essential for production planning, efficiency measurement, and cost control.
        
        **Problem Statement:**  
        Our goal is to predict SMV using various factors such as operation type, yarn type, knit construction, and machine settings. With a focus on optimizing labor productivity and cost estimation, we employ machine learning models like **Random Forest** and **XGBoost** to deliver high-accuracy predictions. These predictions help decision-makers streamline processes, improve time management, and reduce production costs.

        **Approach:**

        - **Data Collection:**  
        We gathered real-world data from textile operations, capturing a variety of critical factors:
            - Operation Type (e.g., sewing, knitting, dyeing)
            - Yarn Type (cotton, polyester, blends, etc.)
            - Knit Construction (patterns and techniques used)
            - Machine Parameters (speed, settings, needle types)
        These variables were selected based on their direct impact on the SMV and operational efficiency.

        - **Data Preprocessing:**  
        Raw data is rarely perfect. To prepare the data for machine learning models, we followed these essential steps:
            - **Cleaning:** Removing irrelevant or redundant data, and addressing missing values.
            - **Handling Outliers:** Identifying extreme values and deciding whether to cap or remove them.
            - **Encoding Categorical Variables:** Transforming categorical features into numerical values through **one-hot encoding** for compatibility with the machine learning models.
            - **Feature Scaling:** Normalizing numerical features like length, speed, and material width to prevent bias during training.

        - **Feature Engineering:**  
        In this step, we enhanced the dataset by creating new, insightful features. Some of the engineered features include:
            - Interaction terms between machine speed and operation type.
            - Transformations (e.g., log, polynomial) of skewed data to improve model fit.
            These new features help the models learn complex relationships between different parameters and improve prediction accuracy.

        - **Model Training and Evaluation:**  
        We used two advanced machine learning algorithms: **Random Forest** and **XGBoost**. Both models are highly effective at handling structured data and offer robust, accurate predictions.
            - **Random Forest:** Builds multiple decision trees, averages their predictions, and prevents overfitting by bagging.
            - **XGBoost:** A highly optimized gradient boosting model that’s efficient, fast, and works well with sparse datasets.
        
        After training, the models are evaluated based on how well they predict SMV values compared to the actual data. Metrics like **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-Squared** are used to measure performance.
        
        - **Model Comparison:**  
        Once trained, we compared both models using evaluation metrics and identified the model that performed best. This model is deployed to provide predictions in real-time in this application.
    """)


with tab3:
   # st.markdown("## Data Preparation: Getting Ready for Modeling")
    st.header("📊 Data Preparation for SMV Prediction")
    st.write("""
        Data preparation is a crucial part of any machine learning project. Here's how we processed our dataset for the best possible results.

        - **Data Collection:**  
        We worked with a rich dataset consisting of multiple parameters that influence SMV. Some key features include:
            - Operation Type: Different tasks performed in textile production.
            - Yarn Type: The materials used in production (e.g., cotton, wool, polyester).
            - Machine Speed: The speed at which the machine operates during the task.
            - Knit Construction: The technique or pattern used in knitting.
            - Material Length: The length of material processed during operations.

        - **Preprocessing Steps:**  
        This phase focuses on cleaning and organizing the raw data:
            - **Data Cleaning:** Eliminating irrelevant or redundant data, filling missing values, and correcting inconsistencies.
            - **Handling Missing Values:** For numerical features, missing values were imputed using statistical methods like the median or mean. For categorical data, the mode was used or the rows were dropped based on the extent of missingness.
            - **Encoding Categorical Variables:** Categorical variables such as operation type, yarn type, and knit construction were converted into numerical format using **one-hot encoding** to allow machine learning models to interpret them.
            - **Scaling Numerical Features:** Features such as material length and machine speed were scaled using **Min-Max Scaling** to ensure that no feature dominates the others due to its range.

        - **Feature Engineering:**  
        To improve the model's predictive power, we engineered additional features:
            - Created interaction terms to capture relationships between features (e.g., operation type × machine speed).
            - Applied transformations like logarithmic scaling to deal with skewed distributions.
            - Generated new features from existing ones to enhance model learning.
    """)

with tab4:
   # st.markdown("## Modeling: Random Forest & XGBoost")
    st.header("💻 Modeling Techniques: Random Forest & XGBoost")
    st.write("""
        We used two powerful machine learning models to predict SMV:

        **Random Forest:**
        - An ensemble method that constructs multiple decision trees and aggregates their predictions.
        - It’s highly accurate for datasets with non-linear relationships.
        - Random Forest also reduces the risk of overfitting by averaging the outcomes of multiple trees.
        - Suitable for datasets with both numerical and categorical variables.

        **XGBoost:**
        - A high-performance implementation of gradient boosting.
        - It combines the predictions of weak learners in an iterative fashion, improving the final prediction step by step.
        - XGBoost is efficient, scalable, and often outperforms other algorithms in structured data scenarios.
        - Incorporates regularization techniques to control model complexity and improve generalization.

        Both models were trained on the same dataset, allowing us to compare their performance on key metrics like **MAE**, **RMSE**, and **R-Squared**. Each model's strengths were assessed, and the most effective one was chosen for deployment.
    """)

with tab5:
   # st.markdown("## Results: Error Analysis & Model Performance")
    st.header("📈 Results: Error Analysis & Model Performance")
    st.write("""
        After training and testing the models, we performed a detailed evaluation to determine how well each model predicted SMV. The key performance metrics include:

        - **Mean Absolute Error (MAE):**  
        Measures the average absolute difference between predicted and actual SMV values.

        - **Mean Squared Error (MSE):**  
        Squares the errors to penalize larger differences between predictions and actual values.

        - **R-Squared:**  
        A statistical measure that represents the proportion of variance in the dependent variable that can be explained by the independent variables. It helps to evaluate the goodness-of-fit for the model.

        **Conclusion:**  
        After comparing the performance of both models on these metrics, we selected the one with the best accuracy and least error for deployment. This model is now used for real-time SMV predictions, improving the overall efficiency of the textile manufacturing process.
    """)

# Footer with acknowledgments
st.markdown("---")
st.write("Developed by [INDESORE](https://www.indesore.com/) - All Rights Reserved © 2024")
