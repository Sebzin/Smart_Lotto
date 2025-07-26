
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import timedelta

st.set_page_config(page_title="Smart Lotto Dashboard", layout="wide")
st.title("🔮 Smart Lotto Prediction & Recurrence Dashboard")

@st.cache_data
def load_data(file_path="lotto_history.xlsx"):
    df = pd.read_excel(file_path)
    numbers_cols = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5']
    df['Draw Date'] = pd.to_datetime(df['Draw Date'])
    df['Sum'] = df[numbers_cols].sum(axis=1)
    df['Mean'] = df[numbers_cols].mean(axis=1)
    df['Median'] = df[numbers_cols].median(axis=1)
    df['StdDev'] = df[numbers_cols].std(axis=1)
    df['Min'] = df[numbers_cols].min(axis=1)
    df['Max'] = df[numbers_cols].max(axis=1)
    df['Range'] = df['Max'] - df['Min']
    df['OddCount'] = df[numbers_cols].apply(lambda row: sum(n % 2 != 0 for n in row), axis=1)
    df['EvenCount'] = df[numbers_cols].apply(lambda row: sum(n % 2 == 0 for n in row), axis=1)
    df['ConsecutivePairs'] = df[numbers_cols].apply(lambda row: sum(1 for a, b in zip(sorted(row), sorted(row)[1:]) if b - a == 1), axis=1)
    df['HotnessScore'] = 0
    return df, numbers_cols

df, number_cols = load_data()
latest_draw = df['Draw Date'].max()

feature_cols = ['Sum', 'Mean', 'Median', 'StdDev', 'Min', 'Max', 'Range',
                'OddCount', 'EvenCount', 'ConsecutivePairs', 'HotnessScore']
X = df[feature_cols]
y1 = df['Number1'].shift(-1).dropna().astype(int)
y2 = df['Number2'].shift(-1).dropna().astype(int)
y3 = df['Number3'].shift(-1).dropna().astype(int)
X = X.iloc[:-1]

enc1, enc2, enc3 = LabelEncoder(), LabelEncoder(), LabelEncoder()
y1_enc, y2_enc, y3_enc = enc1.fit_transform(y1), enc2.fit_transform(y2), enc3.fit_transform(y3)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model1 = XGBClassifier(n_estimators=20, use_label_encoder=False, eval_metric='mlogloss')
model2 = XGBClassifier(n_estimators=20, use_label_encoder=False, eval_metric='mlogloss')
model3 = XGBClassifier(n_estimators=20, use_label_encoder=False, eval_metric='mlogloss')
model1.fit(X_scaled, y1_enc)
model2.fit(X_scaled, y2_enc)
model3.fit(X_scaled, y3_enc)

tab1, tab2 = st.tabs(["🎯 Next Draw Prediction", "📅 When Will It Hit?"])

with tab1:
    st.subheader("Predicted Numbers for the Next Draw After: " + latest_draw.strftime("%Y-%m-%d"))
    last_row = df[feature_cols].iloc[-1:]
    last_scaled = scaler.transform(last_row)
    pred1 = enc1.inverse_transform(model1.predict(last_scaled))[0]
    pred2 = enc2.inverse_transform(model2.predict(last_scaled))[0]
    pred3 = enc3.inverse_transform(model3.predict(last_scaled))[0]
    st.success(f"🎉 Predicted Numbers: {pred1}, {pred2}, {pred3}")

with tab2:
    st.subheader("📅 Estimate When a Number Will Hit Again")

    if 'selected_nums' not in st.session_state:
        st.session_state.selected_nums = []

    st.markdown("### Click numbers to select:")
    cols = st.columns(12)
    for i in range(1, 37):
        if cols[(i - 1) % 12].button(str(i), key=f"btn_{i}"):
            if i in st.session_state.selected_nums:
                st.session_state.selected_nums.remove(i)
            else:
                st.session_state.selected_nums.append(i)

    st.markdown(f"**Selected Numbers:** {st.session_state.selected_nums}")

    def next_probable_dates(df, selected_numbers):
        rows = []
        for number in selected_numbers:
            matches = df[df[number_cols].isin([number]).any(axis=1)].sort_values("Draw Date")
            if len(matches) < 2:
                rows.append((number, "-", "-", "-", "Insufficient Data"))
                continue
            gaps = matches["Draw Date"].diff().dt.days.dropna()
            avg_gap = gaps.mean()
            last_seen = matches["Draw Date"].max()
            est_date = last_seen + timedelta(days=int(round(avg_gap)))
            rows.append((number, round(avg_gap, 1), last_seen.date(), (est_date - latest_draw).days, est_date.date()))
        return pd.DataFrame(rows, columns=["Number", "Avg Gap", "Last Seen", "Days Till Hit", "Probable Date"])

    if st.session_state.selected_nums:
        result_df = next_probable_dates(df, st.session_state.selected_nums)
        st.dataframe(result_df, use_container_width=True)
