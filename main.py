import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

def distribution(df):
    columns = list(df.columns)
    columns.remove("country")
    columns.remove("id")
    hist_column = st.selectbox("Choose column:", tuple(columns))
    if hist_column:
        fig, ax = plt.subplots()
        ax.hist(df.drop(["country"], 1)[hist_column])
        st.pyplot(fig)


def dependency(df):
    columns = list(df.columns)
    columns.remove("country")
    columns.remove("id")
    x = st.selectbox("Choose x axis:", tuple(columns))
    y = st.selectbox("Choose y axis", tuple(columns))
    if x and y:
        fig, ax = plt.subplots()
        ax.scatter(x=df[x], y=df[y])
        st.pyplot(fig)


def fit_model(df):
    dropped_features = st.multiselect("Features to drop:", tuple(df.drop(["life_expectancy"], 1).columns))
    model_type = st.selectbox("Model:", ("Linear regression", "Ridge", "Random forest regressor"))
    model = None
    if model_type == "Linear regression":
        model = LinearRegression()
    elif model_type == "Ridge":
        alpha = st.slider("Regularization", .01, 2.0, value=1.0)
        model = Ridge(alpha=alpha)
    elif model_type == "Random forest regressor":
        n_estimators = st.slider("Number of trees", 10, 200, value=100)
        max_depth = st.slider("Max tree depth", 100, 200)
        min_samples_split = st.slider("The minimum number of samples required to split an internal node", 2, 10, value=2)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=-1)
    if st.button("Evaluate"):
        train = pd.get_dummies(df.drop(dropped_features, 1)).interpolate()
        values = train.drop(["life_expectancy"], 1)
        target = train["life_expectancy"]
        with st.spinner("Evaluating"):
            st.write(cross_validate(model, values, target, scoring="neg_root_mean_squared_error", n_jobs=-1)["test_score"])

def main():
    st.markdown("""
    # Life expectancy prediction app
    """)

    data = pd.read_csv("train.csv")
    data["status"] = data["status"].astype("category")
    mode = st.sidebar.selectbox("Play with the data:", ("Distribution", "Dependencies", "Fit model"))
    if mode == "Distribution":
        distribution(data)
    elif mode == "Dependencies":
        dependency(data)
    elif mode == "Fit model":
        fit_model(data)


if __name__ == '__main__':
    main()