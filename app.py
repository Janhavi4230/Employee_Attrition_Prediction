import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ----------------------------
# Title
# ----------------------------

st.title("👨‍💼 Employee Attrition Prediction")

st.markdown(
    "Predict whether an employee is likely to leave the company."
)

st.divider()

# ===========================
# Employee Details
# ===========================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        18,
        60,
        30
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    distance = st.number_input(
        "Distance From Home",
        1,
        30,
        5
    )

    education = st.selectbox(
        "Education",
        [1,2,3,4,5]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    environment = st.selectbox(
        "Environment Satisfaction",
        [1,2,3,4]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

with col2:

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        50000,
        5000
    )

    overtime = st.selectbox(
        "Over Time",
        [
            "Yes",
            "No"
        ]
    )

    total_working_years = st.number_input(
        "Total Working Years",
        0,
        40,
        10
    )

    years_company = st.number_input(
        "Years At Company",
        0,
        40,
        5
    )

    years_promotion = st.number_input(
        "Years Since Last Promotion",
        0,
        15,
        1
    )

    training = st.number_input(
        "Training Times Last Year",
        0,
        10,
        2
    )

st.divider()

predict = st.button(
    "🔮 Predict Attrition",
    use_container_width=True
)
# ==========================================
# Encoding
# ==========================================

travel_map = {
    "Travel_Rarely": 2,
    "Travel_Frequently": 1,
    "Non-Travel": 0
}

department_map = {
    "Human Resources": 0,
    "Research & Development": 1,
    "Sales": 2
}

education_field_map = {
    "Human Resources": 0,
    "Life Sciences": 1,
    "Marketing": 2,
    "Medical": 3,
    "Other": 4,
    "Technical Degree": 5
}

gender_map = {
    "Female": 0,
    "Male": 1
}

job_role_map = {
    "Healthcare Representative": 0,
    "Human Resources": 1,
    "Laboratory Technician": 2,
    "Manager": 3,
    "Manufacturing Director": 4,
    "Research Director": 5,
    "Research Scientist": 6,
    "Sales Executive": 7,
    "Sales Representative": 8
}

marital_map = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

overtime_map = {
    "No": 0,
    "Yes": 1
}

# ==========================================
# Prediction
# ==========================================

if predict:

    input_dict = {

        "Age": age,

        "BusinessTravel": travel_map[business_travel],

        "Department": department_map[department],

        "DistanceFromHome": distance,

        "Education": education,

        "EducationField": education_field_map[education_field],

        "EnvironmentSatisfaction": environment,

        "Gender": gender_map[gender],

        "JobRole": job_role_map[job_role],

        "MaritalStatus": marital_map[marital],

        "MonthlyIncome": monthly_income,

        "OverTime": overtime_map[overtime],

        "TotalWorkingYears": total_working_years,

        "YearsAtCompany": years_company,

        "YearsSinceLastPromotion": years_promotion,

        "TrainingTimesLastYear": training

    }

    input_df = pd.DataFrame([input_dict])

    # Missing columns add kara
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Order same as training
    input_df = input_df[feature_names]

    # Scaling
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)[0][1]
    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Employee is Likely to Leave the Company")

        st.metric(
            "Attrition Probability",
            f"{probability*100:.2f}%"
        )

    else:

        st.success("✅ Employee is Likely to Stay in the Company")

        st.metric(
            "Retention Probability",
            f"{(1-probability)*100:.2f}%"
        )

    st.progress(float(probability))

    st.divider()

    st.subheader("Employee Details")

    st.dataframe(input_df)    