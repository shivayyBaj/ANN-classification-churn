# 🤖 Customer Churn Prediction AI Dashboard

An interactive **AI-powered Customer Churn Prediction Dashboard** built with **Streamlit, TensorFlow, and Gemini AI**.

This application predicts whether a bank customer is likely to **churn (leave the bank)** and provides an **AI chatbot to explain churn risk, insights, and recommendations** based on the model prediction.

---

## 🚀 Features

- 📊 Customer churn prediction using a **Deep Learning model**
- 📈 Interactive analytics dashboard
- 📉 Churn **risk score visualization**
- 📂 **Batch CSV prediction** for multiple customers
- 🤖 **AI chatbot** for churn explanation
- 📥 Download prediction results as CSV
- 🔐 Secure API key handling using `.env`

---

## 🧠 Tech Stack

- Python
- Streamlit
- TensorFlow
- Scikit-Learn
- Pandas
- Plotly
- Gemini AI API
- python-dotenv

---

## 📂 Project Structure

```
customer-churn-ai
│
├── app.py
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
├── .env   (not pushed to GitHub)
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/customer-churn-ai.git
```

### 2️⃣ Navigate into the project folder

```
cd customer-churn-ai
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 🔑 Setup Environment Variables

Create a `.env` file in the project root folder.

```
GEMINI_API_KEY=your_gemini_api_key
```

This keeps your **API key secure and hidden from GitHub**.

---

## ▶ Run the Application

Start the Streamlit app:

```
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 📊 How It Works

1️⃣ User enters customer information
2️⃣ The trained **Deep Learning model predicts churn probability**
3️⃣ The dashboard visualizes churn risk and analytics
4️⃣ The **AI chatbot explains the churn risk and provides insights**

---

## 📸 Example Features

- Customer churn probability
- Risk score gauge chart
- Churn distribution analytics
- AI-powered explanation of churn predictions
- Batch prediction for CSV datasets

---

## 🎯 Use Cases

- Banking customer retention analysis
- Customer risk monitoring dashboards
- AI-powered business analytics tools
- Machine Learning model deployment demonstration

---

## 👨‍💻 Author

**Shivesh Bajpai**

AI / Machine Learning Enthusiast

GitHub:
https://github.com/your-username

---

## ⭐ If you like this project

Consider giving it a ⭐ on GitHub!
