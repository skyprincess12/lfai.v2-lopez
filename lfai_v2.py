import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(page_title="LFai - Liquidation Factor AI", layout="wide", initial_sidebar_state="expanded")

st.title("📊 LFai – LIQUIDATION FACTOR AI")
st.subheader("Smart Weekly LF Prediction Assistant for Sugar Operations")
st.caption("I Predict. YOU Decide.")

# --- Load Models ---
@st.cache_resource
def load_models():
    ridge_model = joblib.load("models/ridge_model.pkl")
    scaler_model = joblib.load("models/scaler.pkl")
    lgbm_model = lgb.Booster(model_file="models/lgb_model.txt")
    return ridge_model, scaler_model, lgbm_model

ridge, scaler, lgbm = load_models()

feature_cols = ['pol_juice', 'brix_juice', 'purity_juice', 'trs', 'sugar_due',
                'present_stock', 'disruption_time', 'fiber_cane', 'actual_sugar',
                'season_peak', 'season_post_peak', 'season_pre_peak']

input_mode = st.radio("Select Input Mode", ["📤 Upload CSV", "📝 Manual Entry"], horizontal=True)

if input_mode == "📤 Upload CSV":
    uploaded_file = st.file_uploader("Choose your weekly prediction CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    st.markdown("### Manual Entry for One Week")
    date_input = st.date_input("Week Ending Date")
    input_data = {
        'week_ending_date': [date_input.strftime('%Y-%m-%d')],
        'pol_juice': [st.number_input('Pol Juice', step=0.01)],
        'brix_juice': [st.number_input('Brix Juice', step=0.01)],
        'purity_juice': [st.number_input('Purity Juice', step=0.01)],
        'trs': [st.number_input('TRS', step=0.01)],
        'sugar_due': [st.number_input('Sugar Due', step=0.01)],
        'present_stock': [st.number_input('Present Stock', step=0.01)],
        'disruption_time': [st.number_input('Disruption Time', step=0.01)],
        'fiber_cane': [st.number_input('Fiber Cane', step=0.01)],
        'actual_sugar': [st.number_input('Actual Sugar', step=0.01)],
        'season': [st.selectbox("Season", ["season_pre_peak", "season_peak", "season_post_peak"])]
    }
    df = pd.DataFrame(input_data)

if 'df' in locals():
    # One-hot encode season
    for s in ["season_pre_peak", "season_peak", "season_post_peak"]:
        df[s] = (df['season'] == s).astype(int)

    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    ridge_preds = ridge.predict(X_scaled)
    lgb_preds = lgbm.predict(X)
    hybrid_preds = (ridge_preds + lgb_preds) / 2

    apply_bias = st.checkbox("Apply Policy Offset (+2.94%)", value=True)
    df['predicted_lf'] = hybrid_preds.round(2)
    if apply_bias:
        df['adjusted_lf'] = (df['predicted_lf'] + 2.94).round(2)
        df['justification'] = "Bias correction of +2.94% due to CY23–24 policy shift"
    else:
        df['adjusted_lf'] = df['predicted_lf']
        df['justification'] = "No bias applied"

    df_display = df[['week_ending_date', 'predicted_lf', 'adjusted_lf', 'justification']].copy()
    df_display.insert(0, "Week #", range(1, len(df_display)+1))
    df_display['week_ending_date'] = pd.to_datetime(df_display['week_ending_date'], dayfirst=True).dt.strftime('%Y-%m-%d')
    df_display['predicted_lf'] = df_display['predicted_lf'].map("{:.2f}%".format)
    df_display['adjusted_lf'] = df_display['adjusted_lf'].map("{:.2f}%".format)

    st.markdown("### 📋 Weekly Predicted LF")
    st.dataframe(df_display)

    # Append to history
    history_path = "prediction_history.csv"
    if os.path.exists(history_path):
        old_hist = pd.read_csv(history_path)
        combined_hist = pd.concat([old_hist, df_display], ignore_index=True)
    else:
        combined_hist = df_display.copy()
    combined_hist.to_csv(history_path, index=False)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_display['week_ending_date'],
                             y=df['adjusted_lf'],
                             mode='lines+markers',
                             name='Adjusted LF'))
    fig.update_layout(title='📈 Adjusted LF Trend', xaxis_title='Week Ending', yaxis_title='LF (%)', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download LF Predictions", csv, "predicted_lf_report.csv", "text/csv")

# --- Improved Correlation Viewer with Chart Type ---
with st.sidebar.expander("📊 Show Correlation Insight"):
    show_corr = st.checkbox("Enable Correlation Chart", value=False)
    chart_type = st.radio("Chart Type", ["Gradient Bar", "Heatmap"], horizontal=True)

if 'df' in locals() and show_corr:
    st.markdown("### 🔍 Feature Correlation with LF")
    target_col = st.radio("Target LF column", ["predicted_lf", "adjusted_lf"], horizontal=True, key="corr_target_key")
    corr_matrix = df[feature_cols + ['predicted_lf', 'adjusted_lf']].corr()
    corr = corr_matrix[target_col].drop(target_col).sort_values()

    if chart_type == "Gradient Bar":
        fig_corr = px.bar(
            corr,
            orientation='h',
            color=corr,
            color_continuous_scale='RdBu',
            title=f"Correlation with {target_col}",
            labels={'value': 'Correlation'}
        )
        fig_corr.update_layout(xaxis_title='Correlation Coefficient', yaxis_title='Features', template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)

    elif chart_type == "Heatmap":
        fig_heatmap = px.imshow(
            corr_matrix.loc[feature_cols, [target_col]],
            text_auto=True,
            color_continuous_scale='RdBu',
            title=f"Heatmap: Correlation with {target_col}"
        )
        fig_heatmap.update_layout(template='plotly_dark')
        st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Scatterplot Viewer (Top 3 Features by Absolute Correlation) ---
with st.sidebar.expander("📈 Show Feature vs LF Scatterplots"):
    show_scatters = st.checkbox("Enable Scatterplots", value=False)

if 'df' in locals() and show_scatters:
    scatter_target = st.radio("Target LF column", ["predicted_lf", "adjusted_lf"], horizontal=True, key="scatter_target_key")
    st.markdown(f"### 📈 Top 3 Correlated Features (Scatterplots vs {scatter_target})")

    abs_corr = df[feature_cols + ['predicted_lf', 'adjusted_lf']].corr()[scatter_target].drop(scatter_target).abs()
    top_features = abs_corr.sort_values(ascending=False).head(3).index.tolist()

    for col in top_features:
        fig = px.scatter(df, x=col, y=scatter_target,
                         trendline='ols',
                         title=f"{col} vs {scatter_target}",
                         labels={col: col, scatter_target: scatter_target})
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# --- Show History ---
if st.sidebar.checkbox("📜 Show Prediction History"):
    if os.path.exists("prediction_history.csv"):
        hist_df = pd.read_csv("prediction_history.csv")
        st.markdown("### 📜 Prediction History Log")
        st.dataframe(hist_df)
        hist_csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download History", hist_csv, "lf_prediction_history.csv", "text/csv")
    else:
        st.info("No prediction history found.")
