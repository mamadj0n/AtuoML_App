import pandas as pd
import streamlit as st
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.regression import RegressionExperiment
from pycaret.time_series import TSForecastingExperiment
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="AutoML Platform", page_icon="🚀", layout="wide")

st.title("🚀 AutoML Platform with Time Series Forecasting")

# Sidebar for additional settings
with st.sidebar:
    st.header("⚙️ Settings")
    show_data_info = st.checkbox("Show Data Info", value=True)
    random_state = st.number_input("Random State", value=42, min_value=0)

get_data = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if get_data is not None:
    df = pd.read_csv(get_data)
    
    if show_data_info:
        st.write("### 📊 Data Preview:")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    task = st.selectbox(
        "🎯 Select Task", 
        ["Clustering", "Regression", "Classification", "Time Series Forecasting"],
        key="task_select"
    )
    
    # ==================== CLUSTERING ====================
    if task == "Clustering":
        st.header("🔵 Clustering Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Select Clustering Algorithm",
                ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan"],
            )
        with col2:
            num_clusters = st.slider(
                "Number of Clusters", min_value=2, max_value=10, value=4
            )
        
        if st.button("▶️ Run Clustering", key="run_clustering"):
            with st.spinner("Training clustering model..."):
                try:
                    exp = ClusteringExperiment()
                    exp.setup(data=df, html=False)
                    model = exp.create_model(model_type, num_clusters=num_clusters)
                    
                    st.success("✅ Clustering completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Model Summary:")
                        st.write(model)
                    
                    with col2:
                        st.write("### Model Metrics:")
                        metrics = exp.pull()
                        st.dataframe(metrics)
                    
                    st.write("### Data with Cluster Labels:")
                    clustered_df = exp.assign_model(model)
                    st.dataframe(clustered_df.head(10))
                    
                    # Download clustered data
                    csv = clustered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Clustered Data",
                        data=csv,
                        file_name="clustered_data.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ==================== REGRESSION ====================
    elif task == "Regression":
        st.header("📈 Regression Analysis")
        
        target_column = st.selectbox("Select Target Column", df.columns, key="reg_target")
        
        if st.button("▶️ Run Regression", key="run_regression"):
            with st.spinner("Training and comparing regression models..."):
                try:
                    exp = RegressionExperiment()
                    exp.setup(data=df, target=target_column, html=False)
                    
                    best_model = exp.compare_models()
                    
                    st.success("✅ Regression completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Best Regression Model:")
                        st.write(best_model)
                    
                    with col2:
                        st.write("### Model Metrics:")
                        metrics = exp.pull()
                        st.write(metrics.iloc[0])
                    
                    st.write("### Leaderboard:")
                    leaderboard = exp.pull()
                    st.dataframe(leaderboard)
                    
                    # Download results
                    csv = leaderboard.to_csv()
                    st.download_button(
                        label="📥 Download Leaderboard",
                        data=csv,
                        file_name="regression_leaderboard.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ==================== CLASSIFICATION ====================
    elif task == "Classification":
        st.header("🎯 Classification Analysis")
        
        target_column = st.selectbox("Select Target Column", df.columns, key="clf_target")
        
        if st.button("▶️ Run Classification", key="run_classification"):
            with st.spinner("Training and comparing classification models..."):
                try:
                    exp = ClassificationExperiment()
                    exp.setup(data=df, target=target_column, html=False)
                    
                    best_model = exp.compare_models()
                    
                    st.success("✅ Classification completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Best Classification Model:")
                        st.write(best_model)
                    
                    with col2:
                        st.write("### Model Metrics:")
                        metrics = exp.pull()
                        st.write(metrics.iloc[0])
                    
                    st.write("### Leaderboard:")
                    leaderboard = exp.pull()
                    st.dataframe(leaderboard)
                    
                    # Download results
                    csv = leaderboard.to_csv()
                    st.download_button(
                        label="📥 Download Leaderboard",
                        data=csv,
                        file_name="classification_leaderboard.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ==================== TIME SERIES FORECASTING ====================
    elif task == "Time Series Forecasting":
        st.header("📅 Time Series Forecasting")
        
        col1, col2 = st.columns(2)
        with col1:
            date_column = st.selectbox(
                "Select Date/Time Column", 
                df.columns,
                help="Select the column containing dates or timestamps"
            )
        
        with col2:
            target_column = st.selectbox(
                "Select Value Column (Target)", 
                df.columns,
                help="Select the column containing values to forecast"
            )
        
        st.write("### Configuration:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_period = st.slider(
                "Forecast Period",
                min_value=1,
                max_value=30,
                value=10,
                help="Number of future steps to forecast"
            )
        
        with col2:
            frequency = st.selectbox(
                "Data Frequency",
                ["D", "W", "M", "Q", "Y"],
                help="D=Daily, W=Weekly, M=Monthly, Q=Quarterly, Y=Yearly"
            )
        
        with col3:
            train_test_split = st.slider(
                "Train Ratio",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            seasonal_period = st.number_input(
                "Seasonal Period",
                min_value=1,
                value=12,
                help="Number of observations per season"
            )
            cv_folds = st.number_input(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=3
            )
        
        if st.button("▶️ Run Forecasting", key="run_forecasting"):
            with st.spinner("Setting up time series data and training models..."):
                try:
                    df_ts = df.copy()
                    df_ts[date_column] = pd.to_datetime(df_ts[date_column])
                    df_ts = df_ts.sort_values(by=date_column).reset_index(drop=True)
                    
                    # مراحل setup
                    exp = TSForecastingExperiment()
                    exp.setup(
                        data=df_ts,
                        target=target_column,
                        fh=forecast_period,
                        fold=cv_folds,
                        train_size=train_test_split,
                        seasonal_period=seasonal_period,
                        html=False
                    )
                    
                    st.info("🔄 Comparing models...")
                    best_model = exp.compare_models()
                    
                    st.success("✅ Forecasting completed!")
                    
                    # Leaderboard
                    st.write("### Models Leaderboard:")
                    leaderboard = exp.pull()
                    st.dataframe(leaderboard)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Best Model:")
                        st.info(f"Model: {type(best_model).__name__}")
                    
                    with col2:
                        st.write("### Model Performance:")
                        metrics = exp.pull()
                        st.metric("MAE", f"{metrics.iloc[0]['MAE']:.4f}")
                    
                    # Forecast
                    st.write("### Forecast Results:")
                    forecast = exp.predict_model(best_model)
                    st.dataframe(forecast)
                    
                    # Visualization
                    st.write("### Forecast Visualization:")
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df_ts[date_column],
                        y=df_ts[target_column],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    if 'pred_label' in forecast.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=forecast['pred_label'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                    
                    fig.update_layout(
                        title='Time Series Forecast',
                        xaxis_title='Date',
                        yaxis_title=target_column,
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = forecast.to_csv()
                    st.download_button(
                        label="📥 Download Forecast",
                        data=csv,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tips:\n- Ensure date column is in proper date format\n- Data should be sorted by date\n- Target column should contain numeric values")

# Footer
st.markdown("---")
st.markdown("🔧 Built with Streamlit + PyCaret | Time Series powered by statsmodels & scikit-learn")
