import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-3-flash-preview")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🏦 AI Customer Churn Prediction Dashboard")

st.markdown(
"""
Predict whether a **bank customer will churn** using a trained Deep Learning model.
"""
)

# -----------------------------
# Load model and preprocessors
# -----------------------------
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Information")

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider("Age", 18, 92, 35)

credit_score = st.sidebar.number_input(
    "Credit Score", 300, 900, 650
)

balance = st.sidebar.number_input(
    "Balance", 0.0, 250000.0, 50000.0
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary", 0.0, 200000.0, 50000.0
)

tenure = st.sidebar.slider("Tenure", 0, 10, 3)

num_products = st.sidebar.slider(
    "Number of Products", 1, 4, 2
)

has_card = st.sidebar.selectbox(
    "Has Credit Card", [0,1]
)

active_member = st.sidebar.selectbox(
    "Is Active Member", [0,1]
)

# -----------------------------
# Customer Summary
# -----------------------------
col1,col2,col3 = st.columns(3)

col1.metric("Age",age)
col2.metric("Credit Score",credit_score)
col3.metric("Balance",f"${balance:,.0f}")

# -----------------------------
# Prepare input data
# -----------------------------
input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_encoder_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_products],
    "HasCrCard":[has_card],
    "IsActiveMember":[active_member],
    "EstimatedSalary":[estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True),geo_df],
    axis=1
)

input_scaled = scaler.transform(input_data)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
prob = None

if st.button("🔮 Predict Churn"):

    with st.spinner("Running AI Model..."):
        prediction = model.predict(input_scaled)
        prob = prediction[0][0]

    st.subheader("Prediction Result")

    risk_score = int(prob*100)

    st.metric("Customer Risk Score",f"{risk_score}/100")

    if prob>0.5:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Customer Likely to Stay")

    st.write(f"### Churn Probability: {prob:.2%}")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={"text":"Churn Risk %"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"red"},
            "steps":[
                {"range":[0,40],"color":"green"},
                {"range":[40,70],"color":"yellow"},
                {"range":[70,100],"color":"red"}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

    # Probability chart
    chart_df = pd.DataFrame({
        "Outcome":["Stay","Churn"],
        "Probability":[1-prob,prob]
    })

    fig2 = px.bar(
        chart_df,
        x="Outcome",
        y="Probability",
        color="Outcome"
    )

    st.plotly_chart(fig2)

# -----------------------------
# Batch Prediction
# -----------------------------
st.divider()
st.header("📂 Batch Customer Prediction")

uploaded_file = st.file_uploader("Upload CSV",type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # Clean column names
    data.columns = data.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Remove unnecessary columns
    drop_cols = ["RowNumber","CustomerId","Surname","Exited"]

    for col in drop_cols:
        if col in data.columns:
            data = data.drop(col,axis=1)

    # Validate required columns
    required_cols = [
        "CreditScore","Gender","Age","Tenure","Balance",
        "NumOfProducts","HasCrCard","IsActiveMember",
        "EstimatedSalary","Geography"
    ]

    for col in required_cols:
        if col not in data.columns:
            st.error(f"Missing column in CSV: {col}")
            st.stop()

    # Encode Gender
    data["Gender"] = label_encoder_gender.transform(data["Gender"])

    # Encode Geography
    geo_encoded = onehot_encoder_geo.transform(data[["Geography"]]).toarray()

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    data = pd.concat([data.drop("Geography",axis=1),geo_df],axis=1)

    # Scale
    scaled = scaler.transform(data)

    # Predict
    preds = model.predict(scaled)

    data["Churn Probability"] = preds
    data["Risk Level"] = np.where(preds>0.5,"High Risk","Low Risk")

    st.dataframe(data)

    # Risk distribution
    st.subheader("Risk Distribution")

    fig = px.bar(data["Risk Level"].value_counts())

    st.plotly_chart(fig)

    # Histogram
    fig2 = px.histogram(
        data,
        x="Churn Probability",
        nbins=20
    )

    st.plotly_chart(fig2)

    st.download_button(
        "Download Predictions",
        data.to_csv(index=False),
        "churn_predictions.csv"
    )

# -----------------------------
# AI Chatbot
# -----------------------------
st.divider()
st.header("🤖 Customer Churn AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask about customer churn...")

if user_prompt:

    st.session_state.messages.append(
        {"role":"user","content":user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    prompt = f"""
You are a banking data analyst.

Customer Data:
{input_data.to_dict()}

Churn Probability:
{prob}

Answer the user's question clearly.

Question:
{user_prompt}
"""

    response = llm.generate_content(prompt)

    answer = response.text

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)