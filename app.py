import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------- CSS --------------------

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.big-title{
    font-size:52px;
    font-weight:700;
    color:white;
}

.subtitle{
    color:#9ca3af;
    font-size:18px;
}

.metric-card{
    background:#1b1f2a;
    padding:20px;
    border-radius:15px;
    border:1px solid #2b2f3d;
}

.green{
    background:#0f5132;
    color:white;
    padding:15px;
    border-radius:10px;
    font-size:20px;
}

.red{
    background:#842029;
    color:white;
    padding:15px;
    border-radius:10px;
    font-size:20px;
}

footer{
visibility:hidden;
}

</style>
""",unsafe_allow_html=True)

# -------------------- HEADER --------------------

st.markdown(
"""
<div class='big-title'>
📈 Stock Movement Predictor
</div>

<div class='subtitle'>
Machine Learning powered next-day stock direction prediction
</div>
""",
unsafe_allow_html=True
)

st.write("")

# -------------------- SIDEBAR --------------------

st.sidebar.title("Settings")

stock=st.sidebar.selectbox(
"Select Stock",
["AAPL","AMZN","GOOGL","MSFT","TSLA"]
)

# -------------------- DATA --------------------

df=pd.read_csv(f"data/processed/{stock}.csv")

features=[
"Return",
"MA5",
"MA10",
"MA20",
"EMA10",
"RSI",
"Volatility"
]

latest=df[features].iloc[-1:]

model=joblib.load(f"models/{stock}_model.pkl")

prediction=model.predict(latest)[0]

confidence=model.predict_proba(latest)[0].max()*100

# -------------------- METRICS --------------------

col1,col2,col3=st.columns(3)

with col1:
    st.metric(
        "Current Price",
        f"${df['Close'].iloc[-1]:.2f}"
    )

with col2:

    if prediction==1:
        st.metric("Prediction","📈 UP")
    else:
        st.metric("Prediction","📉 DOWN")

with col3:

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

st.divider()

# -------------------- ALERT --------------------

if prediction==1:

    st.markdown(
    f"""
<div class='green'>
📈 Model predicts the stock may move <b>UP</b> next trading day.
</div>
""",
unsafe_allow_html=True)

else:

    st.markdown(
    f"""
<div class='red'>
📉 Model predicts the stock may move <b>DOWN</b> next trading day.
</div>
""",
unsafe_allow_html=True)

st.write("")

# -------------------- CHART --------------------

st.subheader("📊 Closing Price")

st.line_chart(
    df.set_index("Date")["Close"].tail(250),
    use_container_width=True
)

# -------------------- DATA --------------------

st.subheader("Recent Data")

st.dataframe(
    df.tail(10),
    use_container_width=True
)

st.write("")

st.caption("Built using Python • Scikit-learn • Streamlit • Yahoo Finance")