import pandas as pd
import project1_Alona_Gauri_Valeria
X = project1_Alona_Gauri_Valeria.X
y = project1_Alona_Gauri_Valeria.y

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression

from sklearn.neural_network import MLPRegressor

def neural_network(h, X_train, y_train, X_test, y_test):
    ann_model = MLPRegressor(
            hidden_layer_sizes=(h,),
            max_iter=5000,  # Increase the number of iterations
            learning_rate_init=0.001,  # Adjust the learning rate
            solver='adam',  # Try different solvers
            early_stopping=True,  # Enable early stopping
            random_state=42
        )
    ann_model.fit(X_train, y_train)
    y_est = ann_model.predict(X_test)
    est_error = sum((y_est-y_test)**2)/len(y_est)

    return est_error

def test_error(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    test_error = (sum ([(y_pred - y_true)**2 for y_pred, y_true in zip(y_pred_test, y_test)])) / len(y_test)
    return test_error

# REGRESSION PART B (1)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# two_fold_results = pd.DataFrame(columns=['Fold', 'h', 'ann_error', 'lamda', 'ridge_error','baseline_erroe'])
rows = []
# Define the number of folds for both levels of cross-validation
K_outer = K_inner = 10

# Define the range of complexity-controlling parameters
h_values = [1, 5, 15, 30, 45, 60]
lambda_values = [0, 15, 30, 45, 60]

# Initialize the KFold instances for two-level cross-validation
outer_cv = KFold(n_splits=K_outer, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=42)

# Initialize lists to store the performance of each model
baseline_scores = []
ridge_scores = []
ann_scores = []
k = 1

# Outer cross-validation loop
for train_index, test_index in outer_cv.split(X):
    # Split the data into training and test sets for the outer loop
    X_outer_train, X_outer_test = X[train_index], X[test_index]
    y_outer_train, y_outer_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_outer_train = scaler.fit_transform(X_outer_train)

    # If y is a pandas Series, reset the index
    if isinstance(y_outer_train, pd.Series):
        y_outer_train = y_outer_train.reset_index(drop=True)
    if isinstance(y_outer_test, pd.Series):
        y_outer_test = y_outer_test.reset_index(drop=True)

    # Initialize lists to store error value for each hyperparameter setting
    lambda_errors = {}
    for lmbda in lambda_values:
        lambda_errors[lmbda] = []

    ann_errors = {}
    for h in h_values:
        ann_errors[h] = []

    # Inner cross-validation loop (model selection and hyperparameter tuning)
    for inner_train_index, inner_test_index in inner_cv.split(X_outer_train):
        # Split the data into training and validation sets for the inner loop
        X_inner_train, X_inner_test = X_outer_train[inner_train_index], X_outer_train[inner_test_index]
        y_inner_train, y_inner_test = y_outer_train[inner_train_index], y_outer_train[inner_test_index]

        scaler = StandardScaler()
        X_inner_train = scaler.fit_transform(X_inner_train)

        # Train regularized linear regression models with different λ values on the inner training set
        for lmbda in lambda_values:
            ridge_model = Ridge(alpha=lmbda)
            ridge_model.fit(X_inner_train, y_inner_train)
            ridge_error = test_error(ridge_model, X_inner_test, y_inner_test)
            lambda_errors[lmbda].append(ridge_error)

        # Train ANN models with different numbers of hidden units on the inner training set
        for h in h_values:
            # Adjusted ANN model with specified parameters
            ann_error = neural_network(h, X_inner_train, y_inner_train, X_inner_test, y_inner_test)
            ann_errors[h].append(ann_error)

    # Select the best hyperparameters based on the inner fold scores
    best_lambda = min(lambda_errors, key=lambda x: min(lambda_errors[x]))
    best_h = min(ann_errors, key=lambda x: min(ann_errors[x]))

    # Baseline
    baseline_model = LinearRegression()
    baseline_model.fit(X_outer_train, y_outer_train)
    baseline_error = test_error(baseline_model, X_outer_test, y_outer_test)

    # Regularized linear regression with the best λ value
    ridge_model = Ridge(alpha=best_lambda)
    ridge_model.fit(X_outer_train, y_outer_train)
    ridge_model.fit(X_inner_train, y_inner_train)
    ridge_error = test_error(ridge_model, X_inner_test, y_inner_test)

    # ANN with the best number of hidden units
    ann_error = neural_network(h, X_outer_train, y_outer_train, X_outer_test, y_outer_test)

    rows.append({'Fold': k, 'h': best_h, 'ann_error': ann_error, 'lambda': best_lambda, 'ridge_error': ridge_error,
                 'baseline_error': baseline_error})
    k += 1

two_fold_results = pd.DataFrame(rows)

print(two_fold_results)