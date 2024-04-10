"""
Project 2 - 02450

Alona Konstantinova s240400@student.dtu.dk
Gauri Agarwal s236707@student.dtu.dk
Valeria Morra s232962@student.dtu.dk

"""

#Linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

#Importing project 1 to retrieve X and y
import project1_Alona_Gauri_Valeria
X = project1_Alona_Gauri_Valeria.X
y = project1_Alona_Gauri_Valeria.y
data = project1_Alona_Gauri_Valeria.data
data.head()

#Part A (1)
#making sure data is standardized
print("Means of X columns", np.mean(X, axis=0))
print("SDs of X columns", np.std(X, axis=0))
#defining methods to calculate test and train error 

def test_error(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    test_error = (sum ([(y_pred - y_true)**2 for y_pred, y_true in zip(y_pred_test, y_test)])) / len(y_test)
    return test_error

def train_error(model, X_train, y_train):
    y_pred_train = model.predict(X_train)
    train_error = (sum([(y_pred - y_true)**2 for y_pred, y_true in zip(y_pred_train, y_train)])) / len(y_train)
    return train_error

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_rmse = train_error(model, X_train, y_train)
test_rmse = test_error(model, X_test, y_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R^2 Score:", train_r2)
print("Test R^2 Score:", test_r2)

#printing weights
#print(model.coef_)

# Scatter plot for training set
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Actual vs. Predicted (Training)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices (Training Set)')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for testing set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='green', label='Actual vs. Predicted (Testing)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices (Testing Set)')
plt.legend()
plt.grid(True)
plt.show()

#Regression part A (2)

#introducing a regularization parameter lambda, estimating generalization error and calculating test error

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10, shuffle=True)

lambda_values = [0, 0.01, 1, 10, 20, 30, 40, 50]
lambda_gen_errors = {}
lambda_train_errors = {}

for lmbda in lambda_values:
    lambda_gen_errors[lmbda] = []
    lambda_train_errors[lmbda] = []

for train_index, test_index in kf.split(X,y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
    for lmbda in lambda_values:
        model = Ridge(alpha=lmbda)
        model.fit(X_train, y_train)
        lambda_gen_errors[lmbda].append(test_error(model, X_test, y_test))
        lambda_train_errors[lmbda].append(train_error(model, X_train, y_train))

lambda_error_list = [(lmbda, sum(lambda_gen_errors[lmbda]) / 10) for lmbda in lambda_values]
lambda_train_error_list = [(lmbda, sum(lambda_train_errors[lmbda]) / 10) for lmbda in lambda_values]

#plotting generalization error (and test error) as a function of lambda
#8.1.1

lambda_values_plot, errors = zip(*lambda_error_list)
_, train_errors = zip(*lambda_train_error_list)

# Plot lambda values vs generalization errors
plt.figure(figsize=(10, 6))
plt.plot(lambda_values_plot, errors, 'o-', label='Generalization Error', color='skyblue')

# Plot lambda values vs training errors
#plt.plot(lambda_values_plot, train_errors, 'o-', label='Training Error', color='orange')

plt.xlabel('Lambda')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()

Regression Part B (1)

from sklearn.neural_network import MLPRegressor
#defining neural network 
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

# Testing different h_values
from sklearn.model_selection import KFold 

h_values = [1, 3, 5, 10, 15, 30, 60, 70, 80]
df = pd.DataFrame(columns=['h', 'error'])

kf = KFold(n_splits=10, shuffle=True)
rows = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    for h in h_values:
        error = neural_network(h, X_train, y_train, X_test, y_test)
        rows.append({'h': h, 'error': error})

df = pd.DataFrame(rows)
            
#Regression part B (2)

#2-layer validation
K_outer = 10
K_inner = 10
rows = []


# Define the range of complexity-controlling parameters
lambda_values = [0, 0.1, 10, 30, 40, 50]
h_values = [1, 5, 15, 30, 60]

# Initialize the KFold instances for two-level cross-validation
outer_cv = KFold(n_splits=K_outer, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=11)
k_outer = 0

# Outer cross-validation loop
for train_outer_index, test_outer_index in outer_cv.split(X):
    X_train_outer, X_test_outer = X[train_outer_index], X[test_outer_index]
    y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]
    
    k_inner = 1
    
    y_train_outer = y_train_outer.reset_index(drop=True)
    y_test_outer = y_test_outer.reset_index(drop=True)
    
    lambda_in_errors = {lmbda: [] for lmbda in lambda_values} # Initialize dictionary to store errors for each lambda
    ann_in_errors = {h: [] for h in h_values}
    
    for train_inner_index, test_inner_index in inner_cv.split(X_train_outer):
        X_train_inner, X_test_inner = X_train_outer[train_inner_index], X_train_outer[test_inner_index]
        y_train_inner, y_test_inner = y_train_outer[train_inner_index], y_train_outer[test_inner_index]
        
        
        # Train ridge regression models with different lambda values on the inner training set
        for lmbda in lambda_values:
            ridge_model = Ridge(alpha=lmbda)
            ridge_model.fit(X_train_inner, y_train_inner)
            error = test_error(ridge_model, X_test_inner, y_test_inner)
            lambda_in_errors[lmbda].append(error)
        
        for h in h_values:
            error = neural_network(h, X_train_inner, y_train_inner, X_test_inner, y_test_inner)
            ann_in_errors[h].append(error)
        
        k_inner += 1
        #end of inner loop
    
    
    
    lambda_avg_errors = {lmbda: np.mean(errors) for lmbda, errors in lambda_in_errors.items()}
    best_lambda = min(lambda_avg_errors, key=lambda lmbda: lambda_avg_errors[lmbda])
    
    h_avg_errors = {h: np.mean(errors) for h, errors in ann_in_errors.items()}
    best_h = min(h_avg_errors, key=lambda h: h_avg_errors[h])
    
    
    ridge_model = Ridge(alpha=best_lambda)
    ridge_model.fit(X_train_outer, y_train_outer)
    ridge_error = test_error(ridge_model, X_test_outer, y_test_outer)
    
    
    nn_error = neural_network(best_h, X_train_outer, y_train_outer, X_test_outer, y_test_outer)
    
    # Baseline
    baseline_model = LinearRegression()
    baseline_model.fit(X_train_outer, y_train_outer)
    baseline_error = test_error(baseline_model, X_test_outer, y_test_outer)

    #print(f'Final errors of each fold {k_outer}:', error)
    k_outer += 1
    
    rows.append({'k_outer': k_outer, 'h': best_h, 'NN error': nn_error, 'lambda': best_lambda, 'ridge_error': ridge_error, 'baseline': baseline_error})
    
two_layer_results = pd.DataFrame(rows)

#Regression part B (3)
import scipy.stats
import scipy.stats as st

def paired_test(zA, zB):
    # compute confidence interval of model A
    alpha = 0.05

    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    z = zA - zB
    CI = st.t.interval(
        1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
    )  # Confidence interval
    p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return p,CI


print( "p-value of pair: NN and Ridge is", paired_test(two_layer_table['NN error'],two_layer_table['ridge_error'])[0], "and CI is:", paired_test(two_layer_table['NN error'], two_layer_table['ridge_error'])[1])

print( "p-value of pair: NN and Baseline is", 	paired_test(two_layer_table['NN error'], 	two_layer_table['baseline_error'])[0], 
	"and CI is:", paired_test(two_layer_table['NN error'], 	two_layer_table['baseline_error'])[1])

print( "p-value of pair: Ridge and Baseline is", 	paired_test(two_layer_table['ridge_error'], 
   	two_layer_table['baseline_error'])[0], 
      	"and CI is:", paired_test(two_layer_table['ridge_error'], 	two_layer_table['baseline_error'])[1])







#Classification part


X = project1_Alona_Gauri_Valeria.X
y = project1_Alona_Gauri_Valeria.classLabels
classNames = project1_Alona_Gauri_Valeria.classNames
print("Count of each class", y.value_counts())
print("Price intervals as classes: ", classNames)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.dummy import DummyClassifier

import numpy as np
#Importing project 1 to retrieve X and y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Baseline Model
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
baseline_pred = dummy_clf.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("Baseline Accuracy:", baseline_accuracy)

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge

kf = KFold(n_splits=10, shuffle=True)

neighbors = [4,5,6,7,15]
k_gen_errors = {}


for k in neighbors:
    k_gen_errors[k] = []
    

for train_index, test_index in kf.split(X,y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        test_accuracy = 1-knn.score(X_test, y_test)
        k_gen_errors[k].append(test_accuracy)


neighbor_error_list = [(k, sum(k_gen_errors[k]) / 10) for k in neighbors]


k_values_plot, errors = zip(*neighbor_error_list)
_, 

# Plot K values vs generalization errors
plt.figure(figsize=(10, 6))
plt.plot(k_values_plot, errors, 'o-', label='Generalization Error', color='skyblue')

plt.xlabel('K')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()

#Logistic regression
