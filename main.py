import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate


def distribution(df):
    columns = list(df.columns)
    hist_column = st.selectbox("Choose column:", tuple(columns))

    # С помощью if <результат виджета>
    # можно вызвать какой то код
    if hist_column:
        fig, ax = plt.subplots()
        ax.hist(df.drop(["country"], 1)[hist_column])

        # Чтобы нарисовать что то достаточно передать график streamlit
        st.pyplot(fig)


def dependency(df):
    columns = list(df.columns)
    x = st.selectbox("Choose x axis:", tuple(columns))
    y = st.selectbox("Choose y axis", tuple(columns))
    if x and y:
        fig, ax = plt.subplots()
        ax.scatter(x=df[x], y=df[y])
        st.pyplot(fig)


def fit_model(df):
    # multiset позволяет выбрать несколько вещей из списка
    dropped_features = st.multiselect("Features to drop:", tuple(df.drop(["life_expectancy"], 1).columns))

    model_type = st.selectbox("Model:", ("Linear regression", "Ridge", "Random forest regressor"))
    model = None

    if model_type == "Linear regression":
        model = LinearRegression()
    elif model_type == "Ridge":
        # value - значение по умолчанию
        alpha = st.slider("Regularization", .01, 2.0, value=1.0)
        model = Ridge(alpha=alpha)
    elif model_type == "Random forest regressor":
        n_estimators = st.slider("Number of trees", 10, 200, value=100)
        max_depth = st.slider("Max tree depth", 100, 200)
        min_samples_split = st.slider("The minimum number of samples required to split an internal node", 2, 10,
                                      value=2)
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      n_jobs=-1)

    # Создаем кнопку и когда на нее нажмут код ниже выполнится
    if st.button("Evaluate"):
        train = pd.get_dummies(df.drop(dropped_features, 1)).interpolate()
        values = train.drop(["life_expectancy"], 1)
        target = train["life_expectancy"]
        # Чтобы пользователь не гадал работает скрипт или нет можно сделать плашку загрузки
        with st.spinner("Evaluating"):
            st.write(
                cross_validate(model, values, target, scoring="neg_root_mean_squared_error", n_jobs=-1)["test_score"])


def main():
    # Можем писать в формате markdown
    st.markdown("""
    # Life expectancy prediction app
    """)

    # Загружаем данные и заменяем
    data = pd.read_csv("train.csv")
    data["status"] = data["status"].astype("category")

    # Добавляем виджет в sidebar. Selectbox выведет ту строку которую
    # выбрал пользователь или None если еще ничего не выбрал
    modes = ("Distribution", "Dependencies", "Fit model")
    mode = st.sidebar.selectbox("Play with the data:", modes)

    # Проверяем вкладку и вызываем соответствующую функцию
    if mode == "Distribution":
        distribution(data.drop(["country", "id"], 1))
    elif mode == "Dependencies":
        dependency(data.drop(["country", "id"], 1))
    elif mode == "Fit model":
        fit_model(data)


if __name__ == '__main__':
    main()
