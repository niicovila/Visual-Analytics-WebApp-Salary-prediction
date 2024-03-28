import streamlit as st

def main():
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

    # Main title and introduction
    st.title("🚀 Salary Insights Tool 🚀")
    st.subheader("Empowering Your IT Workforce Decisions")
    st.write("At our company, we understand the pivotal role that competitive salaries play in attracting and retaining top-tier software development talent. To address this need, we've harnessed the power of data and technology to bring you an innovative Salary Insights Web App.")

    # Key features section
    st.subheader("Key Features:")
    st.markdown("- **Discover Salaries Worldwide**: Explore salaries across the globe using data from the Stack Overflow Annual Survey.")
    st.markdown("- **Predictive Modeling**: Our Machine Learning models predicts salaries based on country, education, and experience.")

    # How to Use section
    st.header("How to Use the Web App:")
    st.subheader("1. Explore Salary Trends:")
    st.write("Navigate to the Data Exploration Tab and dive into the detailed Exploratory Data Analysis (EDA). Gain a comprehensive understanding of salary distributions, regional patterns, and key factors influencing compensation.")

    st.subheader("2. Predict Salaries:")
    st.write("Head over to the Salary Predictions tab to make precise salary predictions. Input specific details such as country, education level, and years of experience to receive personalized insights. The app utilizes different ML models to enhance your user experience.")

if __name__ == "__main__":
    main()
