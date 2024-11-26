import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import mean_squared_error, accuracy_score


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    lasso = Lasso()
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression(penalty=None, max_iter=10000)
    model.fit(X, y)
    return model


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
    model = LogisticRegression(penalty='l2', max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
    model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def test_functions():
    """
    Tests all regression and classification functions.
    """
    print("=== Testing Regression Models ===")
    X_train, X_test, y_train, y_test = get_regression_data()

    lin_reg = linear_regression(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lin, squared=False))

    ridge_reg = ridge_regression(X_train, y_train)
    y_pred_ridge = ridge_reg.predict(X_test)
    print("Ridge Regression RMSE:", mean_squared_error(y_test, y_pred_ridge, squared=False))

    lasso_reg = lasso_regression(X_train, y_train)
    y_pred_lasso = lasso_reg.predict(X_test)
    print("Lasso Regression RMSE:", mean_squared_error(y_test, y_pred_lasso, squared=False))

    print("\n=== Testing Classification Models ===")
    X_train, X_test, y_train, y_test = get_classification_data()

    log_reg = logistic_regression(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
    
    log_reg_l2 = logistic_l2_regression(X_train, y_train)
    y_pred_log_l2 = log_reg_l2.predict(X_test)
    print("Logistic Regression (L2) Accuracy:", accuracy_score(y_test, y_pred_log_l2))

    log_reg_l1 = logistic_l1_regression(X_train, y_train)
    y_pred_log_l1 = log_reg_l1.predict(X_test)
    print("Logistic Regression (L1) Accuracy:", accuracy_score(y_test, y_pred_log_l1))


test_functions()


"""
    Аналіз результатів тестів моделей регресії та класифікації.

    Результати:

    === Моделі регресії ===
    1. Лінійна регресія:
       - RMSE: 57.354
       - Базова модель без регуляризації. Вона може давати гідні результати для задач, де ознаки не мають сильної кореляції.
    2. Ridge-регресія:
       - RMSE: 57.267
       - Дещо краща за лінійну регресію завдяки регуляризації L2. Ця модель штрафує великі коефіцієнти, 
       що робить її більш стійкою до мультиколінеарності.
    3. Lasso-регресія:
       - RMSE: 57.619
       - Показала трохи гірший результат порівняно з Ridge-регресією. Це може бути пов'язано з тим, що Lasso 
       "обнуляє" деякі коефіцієнти, що зменшує гнучкість моделі.

    === Моделі класифікації ===
    1. Логістична регресія (без регуляризації):
       - Accuracy: 0.974
       - Базова модель показала дуже гарні результати, підтверджуючи, що дані добре підходять для класифікації.
    2. Логістична регресія з L2-регуляризацією:
       - Accuracy: 0.974
       - Результати аналогічні моделі без регуляризації. Це свідчить про те, що регуляризація не значно впливає, можливо, через якість даних.
    3. Логістична регресія з L1-регуляризацією:
       - Accuracy: 0.982
       - Найкраща точність серед усіх моделей. Це свідчить про те, що L1-регуляризація покращила роботу, можливо, 
       за рахунок виключення менш важливих ознак.

    === Висновки ===
    1. Для задач регресії:
       - Ridge-регресія показала кращий результат серед моделей завдяки регуляризації L2.
       - Лінійна регресія залишається гідним вибором, якщо обчислювальна простота є пріоритетом.
       - Lasso-регресія може бути корисною для вибору ознак, але в даному випадку це не привело до покращення RMSE.

    2. Для задач класифікації:
       - Логістична регресія з L1-регуляризацією показала найкращу точність (0.982), що робить її оптимальним вибором для цієї задачі.
       - Моделі без регуляризації або з L2-регуляризацією також показали високі результати, тому їх можна розглядати для простіших задач.

    Рекомендації:
       - Для регресійних задач рекомендується використовувати Ridge-регресію.
       - Для класифікаційних задач варто звернути увагу на L1-регуляризацію, особливо якщо важливий вибір ознак.
    """