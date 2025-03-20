import streamlit as st
import pandas as pd
import requests

# 🌟 Set Page Config FIRST (Fixing Error)
st.set_page_config(
    page_title="AI-Powered Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# 🎯 API Endpoint
API_URL = "http://127.0.0.1:5000/predict"

# 📌 Expected Feature Names
FEATURE_NAMES = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length",
    "Total Spend", "Last Interaction", "Engagement Score"
]

# 🎨 Custom CSS for Floating UI
st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 15px rgba(255, 255, 255, 0.1);
        }
        .floating-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 0px 15px rgba(0, 255, 255, 0.5);
        }
        .prediction-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 0px 15px rgba(255, 165, 0, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 App Title
st.markdown("<h1 style='text-align: center; color: cyan;'>📊 AI-Powered Customer Churn Prediction </h1>", unsafe_allow_html=True)

# 📌 Instructions Panel
with st.expander("📢 How to Use? Click Here!"):
    st.markdown("""
        ✅ **Step 1:** Upload a CSV file with these **columns**:
        - `Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction, Engagement Score`
        
        ✅ **Step 2:** Click on "📂 Upload CSV File" and select your file.
        
        ✅ **Step 3:** Click "🔮 Predict Churn" to get AI-based predictions.  
        
        ✅ **Step 4:** Predictions will be displayed in a futuristic **floating UI**.  
    """)

# 📂 File Upload Section
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"], help="Upload a CSV file with correct feature names")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ✅ Check for Missing Columns
    missing_columns = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing columns in uploaded file: {missing_columns}")
    else:
        # 📊 Data Preview in Floating UI
        st.markdown("<h3 style='text-align: center;'>📂 Uploaded Data Preview</h3>", unsafe_allow_html=True)
        st.markdown('<div class="floating-box">', unsafe_allow_html=True)
        st.dataframe(df)
        st.markdown('</div>', unsafe_allow_html=True)

        # 🔮 Predict Button
        if st.button("🔮 Predict Churn", use_container_width=True):
            try:
                response = requests.post(API_URL, json={"features": df.to_dict(orient="records")})
                result = response.json()

                if "churn_predictions" in result:
                    df["Churn Prediction"] = result["churn_predictions"]
                    
                    # 🎯 Display Predictions
                    st.markdown("<h3 style='text-align: center;'>🔍 AI Predictions</h3>", unsafe_allow_html=True)
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error("⚠️ API Error: " + str(result))
            except Exception as e:
                st.error(f"⚠️ Failed to connect to API: {e}")

# 🚀 Footer
st.markdown("<h4 style='text-align: center; color: cyan;'>🚀 Built with **Streamlit & Flask** | Developer: YASH DAVKHAR 🔍📊</h4>", unsafe_allow_html=True)
