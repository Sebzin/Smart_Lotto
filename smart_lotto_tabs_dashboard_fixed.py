
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Smart Split Predictor", layout="wide")
st.title("🎯 Smart Lotto: Prediction Dashboard")

# Load model and encoder
model = joblib.load("ml_split_pair_model.pkl")
encoder = joblib.load("ml_label_encoder_split_pair.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("lotto_history.xlsx")
    df['Draw Date'] = pd.to_datetime(df['Draw Date'])
    number_cols = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5']
    df['DateSum'] = df['Draw Date'].apply(lambda x: x.day + (x.year % 100))
    df['SortedNumbers'] = df[number_cols].apply(lambda row: sorted(row), axis=1)
    df['4thHighest'] = df['SortedNumbers'].apply(lambda x: x[3])
    df['Deferred'] = (df['DateSum'] - df['4thHighest']).abs()
    return df, number_cols

df, number_cols = load_data()

# Draw Date Selector (dropdown)
draw_dates = df['Draw Date'].dt.strftime("%Y-%m-%d").tolist()
selected_str = st.selectbox("📅 Select a Draw Date", draw_dates, index=len(draw_dates)-1)
selected_date = pd.to_datetime(selected_str)

selected_row = df[df['Draw Date'] == selected_date]
if selected_row.empty:
    st.warning("Selected date not found in data.")
    st.stop()

row = selected_row.iloc[0]
datesum = int(row['DateSum'])
deferred = int(row['Deferred'])

st.markdown(f"### 📅 Draw Date: `{selected_str}`")
st.markdown(f"- 🔢 **DateSum:** {datesum}")
st.markdown(f"- 🧮 **Deferred:** {deferred}")

# Generate all valid pairs that sum or differ to target
def generate_split_features(target):
    pairs = [(a, b) for a in range(1, 37) for b in range(a + 1, 37)]
    records = []
    for a, b in pairs:
        pair_sum = a + b
        pair_diff = abs(a - b)
        if pair_sum == target or pair_diff == target:
            records.append({
                'Deferred': target,
                'A': a,
                'B': b,
                'PairSum': pair_sum,
                'PairDiff': pair_diff
            })
    return pd.DataFrame(records)

# Predict top N split pairs and show results
def predict_top_splits(value, label):
    df_feat = generate_split_features(value)
    if df_feat.empty:
        return st.warning(f"No valid splits for {label} value {value}")
    pred = model.predict(df_feat)
    prob = model.predict_proba(df_feat)
    df_feat['Prediction'] = encoder.inverse_transform(pred)
    df_feat['Confidence'] = prob.max(axis=1)
    df_feat['Pair'] = df_feat.apply(lambda row: f"({row.A}, {row.B})", axis=1)
    top_df = df_feat.sort_values("Confidence", ascending=False).head(5)
    st.dataframe(top_df[['Pair', 'Prediction', 'Confidence']].style.format({'Confidence': '{:.2%}'}), use_container_width=True)

# Tabbed layout
tab1, tab2 = st.tabs(["📅 DateSum Prediction", "📅 Deferred Prediction"])
with tab1:
    predict_top_splits(datesum, "DateSum")
with tab2:
    predict_top_splits(deferred, "Deferred")
