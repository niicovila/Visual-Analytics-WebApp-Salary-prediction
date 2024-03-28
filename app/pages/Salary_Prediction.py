import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
CURRENT_PATH = os.path.dirname(__file__)

st.set_page_config(
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",  # Adjust this based on your sidebar preference
    #margin_left=50  # Set the desired left margin in pixels
)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to top, #00008b, #4b0082);
        opacity: 0.7;
        color: white;
    }
    [data-testid="stSidebar"] .sidebar-content {
        
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data()
def load_data():
    file_path = os.path.join(CURRENT_PATH, 'data_processed.csv')
    df = pd.read_csv(file_path)
    return df

# Function to load models and label encoders
@st.cache_data()
def load_models():
    file_path = os.path.join(CURRENT_PATH, 'saved_models.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data["model_rf_best"], data["model_linreg"], data["model_nn"], data["model_gbm"], data["model_lgbm"], \
           data["le_country"], data["le_education"]

# Load the DataFrame
#df = pd.read_csv("/Users/nicolasvila/workplace/uni/Lab10_students (2)/app/data_processed.csv")

# Load models and label encoders
df = load_data()
regressor_loaded, linreg_loaded, nn_loaded, gbm_loaded, lgbm_loaded, le_country, le_education = load_models()


# Get unique countries and education levels from the DataFrame
all_countries = df['Country'].unique()
all_education_levels = df['EdLevel'].unique()
# Create or load dataframe to store predictions
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = pd.DataFrame(columns=['Country', 'EdLevel', 'YearsCodePro', 'Model', 'Predicted Salary'])


# Streamlit UI
st.title("💰 Salary Prediction")
st.header("Introduction to our models")
st.write("We have developed 3 different models, each of them with different capabilities. The model metrics can be found below. The models we have developed are:")
st.markdown("- **Linear Regression**: A simple and interpretable linear modeling technique for predicting numerical outcomes based on linear relationships between input features and the target variable.")
st.markdown("- **Random Forest**: An ensemble learning method that builds multiple decision trees during training and outputs the average prediction (regression) or majority vote (classification) of the individual trees.")
st.markdown("- **Gradient Boosting Regressor**: A machine learning algorithm that builds a predictive model in a stage-wise fashion, combining the predictions of weak models (typically decision trees) to improve accuracy.")
st.markdown("- **Feed Forward Neural Network**: A type of artificial neural network where information travels in one direction, from input nodes through hidden layers to output nodes, making it a fundamental architecture for deep learning.")
st.markdown("- **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms, designed for efficient and distributed training, making it particularly effective for large datasets.")

st.subheader('Training Split')
st.write("In the plot below, we can see the training and test split that was used in order to train the models. Precisely, it is a 80-20 split")
image_path = os.path.join(CURRENT_PATH, 'train_test.png')
st.image(image_path, caption="Training and Test Split", width=600)

st.subheader("Different Models' Metrics on Test Data")
st.write("In the plot below, we plot different metrics such as Mean Squared Error, Mean Absolute Error, and R2. We can see that the models with the best performance in our test data were Gradient Boost adn LightGBM. We can also see two different Random Forest models, that is because one was implemented naively and the other one was implemented with optimal hyperparameters, found by GridSearch, and it is the one at use in this app.")
# Set up the layout with two columns
col1, col2 = st.columns(2)

# Load and display the first image
image_path1 = os.path.join(CURRENT_PATH, 'metrics_models.png')
col1.image(image_path1, caption="Metrics of the used models", use_column_width=True)

# Load and display the second image
image_url2 = os.path.join(CURRENT_PATH, 'r2_models.png')
col2.image(image_url2, caption="R2 metric", use_column_width=True)

st.subheader("Model Discussion")
st.write("Linear Regression is a straightforward and interpretable choice for predicting salaries, particularly suitable when the relationship between predictors and salary is linear. However, it might struggle to capture more intricate patterns, which seems to be our case. Random Forest, known for handling non-linear relationships well, is robust to overfitting and can adapt to unbalanced datasets. Gradient Boosting Regressor, offering high predictive accuracy, can also handle unbalanced datasets but requires careful tuning to prevent overfitting. Feed Forward Neural Networks, powerful for complex patterns, may be suitable for larger datasets but demand meticulous parameter tuning. In simpler cases, it might not perform well. LightGBM, efficient and scalable, stands out for unbalanced datasets due to its ability to handle skewed distributions effectively, making it a favorable choice for predicting salaries with varying class frequencies. The selection among these models hinges on factors like dataset characteristics, interpretability, and computational considerations.")


# Model Selection
st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.radio("Choose a Model", ["Random Forest", "Linear Regression", "Neural Network", "Gradient Boosting Regressor", "LightGBM"])



# User Input
st.subheader("Enter Developer Information:")
country = st.selectbox("Country", all_countries)
education_level = st.selectbox("Education Level", all_education_levels)
years_of_experience = st.slider("Years of Professional Coding Experience", 0, 50, 10)

# Make a prediction based on user input and selected model
if st.button("Predict Salary"):
    if ((st.session_state.predictions_df['Country'] == country) & 
        (st.session_state.predictions_df['EdLevel'] == education_level) & 
        (st.session_state.predictions_df['YearsCodePro'] == years_of_experience) & 
        (st.session_state.predictions_df['Model'] == selected_model)).any():
        st.warning("Prediction with these parameters already exists.")
    # Prepare input data
    else:
        new_data = pd.DataFrame({'Country': [country], 'EdLevel': [education_level], 'YearsCodePro': [years_of_experience]})
        new_data['Country'] = le_country.transform(new_data['Country'])
        new_data['EdLevel'] = le_education.transform(new_data['EdLevel'])

        # Choose the selected model
        if selected_model == "Random Forest":
            predicted_salary = regressor_loaded.predict(new_data)[0]
        elif selected_model == "Linear Regression":
            predicted_salary = linreg_loaded.predict(new_data)[0]
        elif selected_model == "Neural Network":
            predicted_salary = nn_loaded.predict(np.array(new_data)).flatten()[0]
        elif selected_model == "Gradient Boosting Regressor":
            predicted_salary = gbm_loaded.predict(new_data)[0]
        elif selected_model == "LightGBM":
            predicted_salary = lgbm_loaded.predict(new_data)[0]
    
        # Update predictions dataframe
        new_row = pd.DataFrame({
            'Country': [country],
            'EdLevel': [education_level],
            'YearsCodePro': [years_of_experience],
            'Model': [selected_model],
            'Predicted Salary': [predicted_salary]
        })
        st.session_state.predictions_df = pd.concat([st.session_state.predictions_df, new_row], ignore_index=True)

        st.success(f"Predicted Salary using {selected_model}: ${predicted_salary:,.2f}")


# Additional Insights
st.subheader("Additional Insights:")
# Add more features and insights based on predictions (e.g., recommendation, salary distribution)
# ...

# Display the app
st.sidebar.markdown("### About")
st.sidebar.info("This app predicts the salary of software developers based on their country, education, and experience.")

# Display predictions dataframe
st.subheader("Predicted Salaries:")
st.write(st.session_state.predictions_df)