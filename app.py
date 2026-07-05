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
        if st.button("Run Clustering"):
            st.write("Analyzing and comparing clustering models... Please wait.")

            # در پایکارت ۴، تنظیمات با چسباندن .fit() به سازنده کلاس انجام می‌شود
            exp = ClusteringExperiment().fit(df)

            # طبق داکیومنت نسخه ۴، متد compare_models برای همه تسک‌ها یکسان است
            compare_result = exp.compare_models()

            st.write("### Best Clustering Pipeline:")
            st.write(compare_result.best)  # گرفتن پایپ‌لاین بهینه از فیلد best

            st.write("### Leaderboard:")
            st.dataframe(compare_result.leaderboard)

    # ------------------ REGRESSION TASK ------------------
    elif task == "Regression":
        target_column = st.selectbox("Select Target Column", df.columns)

        if st.button("Run Regression"):
            st.write("Training and comparing regression models... Please wait.")

            # تعیین تارگت و فیت کردن اولیه دیتابیس در سازنده کلاس
            exp = RegressionExperiment(target=target_column).fit(df)

            # اجرای مقایسه مدل‌ها
            compare_result = exp.compare_models()

            st.write("### Best Regression Model:")
            st.write(compare_result.best)

            st.write("### Leaderboard:")
            st.dataframe(compare_result.leaderboard)

    # ------------------ CLASSIFICATION TASK ------------------
    elif task == "Classification":
        target_column = st.selectbox("Select Target Column", df.columns)

        if st.button("Run Classification"):
            st.write("Training and comparing classification models... Please wait.")

            # ساخت آزمایش طبق ساختار جدید پایکارت ۴
            exp = ClassificationExperiment(target=target_column).fit(df)

            compare_result = exp.compare_models()

            st.write("### Best Classification Model:")
            st.write(compare_result.best)

            st.write("### Leaderboard:")
            st.dataframe(compare_result.leaderboard)
