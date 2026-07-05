import pandas as pd
import streamlit as st
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.regression import RegressionExperiment

st.set_page_config(page_title="AutoML", page_icon="🚢", layout="wide")

get_data = st.file_uploader("Upload your CSV file", type=["csv"])

if get_data is not None:
    df = pd.read_csv(get_data)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    task = st.selectbox("Select Task", ["Clustering", "Regression", "Classification"])

    # ------------------ CLUSTERING TASK ------------------
    if task == "Clustering":
        # در نسخه 3.x کلاسترینگ متد compare_models ندارد، پس کاربر باید مدل را انتخاب کند
        model_type = st.selectbox(
            "Select Clustering Algorithm",
            ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan"],
        )
        num_clusters = st.slider(
            "Number of Clusters", min_value=2, max_value=10, value=4
        )

        if st.button("Run Clustering"):
            st.write("Training clustering model... Please wait.")
            
            exp = ClusteringExperiment()
            exp.setup(data=df, html=False) # استفاده از setup به سبک نسخه 3.x

            model = exp.create_model(model_type, num_clusters=num_clusters)
            st.write("### Model Summary:")
            st.write(model)

            st.write("### Data with Cluster Labels:")
            clustered_df = exp.assign_model(model)
            st.dataframe(clustered_df.head())

    # ------------------ REGRESSION TASK ------------------
    elif task == "Regression":
        target_column = st.selectbox("Select Target Column", df.columns)

        if st.button("Run Regression"):
            st.write("Training and comparing regression models... Please wait.")
            
            exp = RegressionExperiment()
            exp.setup(data=df, target=target_column, html=False) # بدون پارامتر silent
            
            best_model = exp.compare_models()
            
            st.write("### Best Regression Model:")
            st.write(best_model)
            
            # در نسخه 3.x برای دریافت جدول مقایسه از pull استفاده می‌کنیم
            st.write("### Leaderboard:")
            st.dataframe(exp.pull())

    # ------------------ CLASSIFICATION TASK ------------------
    elif task == "Classification":
        target_column = st.selectbox("Select Target Column", df.columns)

        if st.button("Run Classification"):
            st.write("Training and comparing classification models... Please wait.")
            
            exp = ClassificationExperiment()
            exp.setup(data=df, target=target_column, html=False)
            
            best_model = exp.compare_models()
            
            st.write("### Best Classification Model:")
            st.write(best_model)
            
            st.write("### Leaderboard:")
            st.dataframe(exp.pull())
