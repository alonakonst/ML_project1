"""
Project 1 - 02450

Alona Konstantinova s240400@student.dtu.dk
Gauri Agarwal s236707@student.dtu.dk
Valeria Morra s232962@student.dtu.dk

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel
from sklearn.preprocessing import LabelEncoder
from scipy.stats import normaltest
from scipy.linalg import svd

data=pd.read_csv('diamonds.csv')

#data quality
data['carat'] = pd.to_numeric(data['carat'], errors='coerce')
data['depth'] = pd.to_numeric(data['depth'], errors='coerce')
data['table'] = pd.to_numeric(data['table'], errors='coerce')
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data['z'] = pd.to_numeric(data['z'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')

for col in ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']:
    if data[data[col] < 0].shape[0] > 0:
        print(f"Corrupted data in {col}: ", data[data[col] < 0].shape[0])

for col in ['carat', 'x', 'y', 'z']:
    if data[data[col] == 0].shape[0] > 0:
        print(f"Corrupted data in {col}: ", data[data[col] == 0].shape[0])

expected_cuts = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
expected_colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
expected_clarity = ['SI1', 'VS2', 'SI2', 'VS1', 'VVS2', 'VVS1', 'IF', 'I1']

if not set(data['cut']).issubset(expected_cuts):
    print("Unexpected values in 'cut'")
if not set(data['color']).issubset(expected_colors):
    print("Unexpected values in 'color'")
if not set(data['clarity']).issubset(expected_clarity):
    print("Unexpected values in 'clarity'")

#summary statistics
print(data[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']].describe())

#bar charts for non-numerical attributes
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
cut_counts = data['cut'].value_counts()
axs[0].bar(x=range(len(cut_counts)), height=cut_counts, color='darkred')
axs[0].set_title('cut')
axs[0].set_xticks(range(len(cut_counts)))  # Set the tick positions
axs[0].set_xticklabels(cut_counts.index, rotation=90)  # Set the tick labels with rotation
color_counts = data['color'].value_counts()
axs[1].bar(x=range(len(color_counts)), height=color_counts, color='darkred')
axs[1].set_title('color')
axs[1].set_xticks(range(len(color_counts)))  # Set the tick positions
axs[1].set_xticklabels(color_counts.index, rotation=90)  # Set the tick labels with rotation
clarity_counts = data['clarity'].value_counts()
axs[2].bar(x=range(len(clarity_counts)), height=clarity_counts, color='darkred')
axs[2].set_title('clarity')
axs[2].set_xticks(range(len(clarity_counts)))  # Set the tick positions
axs[2].set_xticklabels(clarity_counts.index, rotation=90)  # Set the tick labels with rotation
plt.savefig('categorical.png')
plt.show()

#box plots for numerical data:
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

axs[0].boxplot(data['x'])
axs[0].set_title('X')
axs[0].set_xticklabels([])

axs[1].boxplot(data['y'])
axs[1].set_title('Y')
axs[1].set_xticklabels([])

axs[2].boxplot(data['z'])
axs[2].set_title('Z')
axs[2].set_xticklabels([])
plt.savefig('boxplot_xyz.png')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 10))

axs[0].boxplot(data['carat'])
axs[0].set_title('Carat')
axs[0].set_xticklabels([])

axs[1].boxplot(data['depth'])
axs[1].set_title('Depth')
axs[1].set_xticklabels([])

axs[2].boxplot(data['table'])
axs[2].set_title('Table')
axs[2].set_xticklabels([])

axs[3].boxplot(data['price'])
axs[3].set_title('Price')
axs[3].set_xticklabels([])
plt.savefig('boxplot_other.png')
plt.show()

#histograms of numerical data

selected_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
df = pd.DataFrame({col: data[col] for col in selected_columns})
nrows = 2
ncols = 4
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 7))
axs = axs.flatten()
for i, column in enumerate(df.columns):
    ax = axs[i]
    ax.hist(df[column], bins=13, color='darkred', alpha=0.7)
    ax.set_title(column.capitalize())  # Set the title to the column name
for ax in axs[len(df.columns):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('hist_overview.png')
plt.show()

#normality test
attributes = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
alpha = 0.05
for attribute in attributes:
    statistic, p_value = normaltest(data[attribute])

    if p_value > alpha:
        print(f"{attribute.capitalize()}: fail to reject H0")
    else:
        print(f"{attribute.capitalize()}: reject H0")


#correlation matrix
cormat = data.corr(numeric_only = True)
round(cormat,2)


#numerical values of 'price' are transformed into 'price_category', which takes values from 0 to 10
num_categories = 10
max_price = data['price'].max()
custom_bin_edges = np.linspace(0, max_price, num=num_categories + 1)
custom_bin_edges = np.unique(custom_bin_edges)
custom_bin_edges[-1] = np.inf
labels = range(len(custom_bin_edges) - 1)
data['price_category'] = pd.cut(data['price'], bins=custom_bin_edges, labels=labels, include_lowest=True)


#Dictionary of the price classes and price intervals
classLabels = data['price_category']
classNames=[]
classDict = {}
C=0
for category, interval in enumerate(pd.cut(data['price'], bins=custom_bin_edges).unique()):
    interval_value = str(interval)
    interval_value = interval_value.replace('(', '').replace('[', '').replace(',', ' -').replace(']', '').replace(')', '')  # Remove parentheses and brackets
    classDict[interval_value]=category
    classNames.append(interval_value)
    C=C+1

y = np.asarray(classLabels)

#Creating X data matrix
x_attributes = [0,1,2,3,4,5,7,8,9]
X = data.iloc[:, x_attributes].values
attributeNames = data.columns[x_attributes].tolist()


#Encoding non-numerical ordinal data into integers: 'cut' attribute
cut_values = X[:, 1]
cut_mapping = {
    'Fair': 0,
    'Good': 1,
    'Very Good': 2,
    'Premium': 3,
    'Ideal': 4
}
mapping_func = np.vectorize(lambda x: cut_mapping[x])

# Convert ordinal values to numerical values
cut_values = X[:, 1]
cut_mapping = {
    'Fair': 0,
    'Good': 1,
    'Very Good': 2,
    'Premium': 3,
    'Ideal': 4
}
mapping_func = np.vectorize(lambda x: cut_mapping[x])
cut_numerical = mapping_func(cut_values)
X[:, 1] = cut_numerical

#Encoding non-numerical ordinal data into integers: 'clarity' attribute
clarity_values = X[:,3]
clarity_mapping = {
    'I1': 0,
    'SI2': 1,
    'SI1': 2,
    'VS2': 3,
    'VS1': 4,
    'VVS2': 5,
    'VVS1': 6,
    'IF': 7,
}
mapping_func = np.vectorize(lambda x: clarity_mapping[x])
clarity_numerical = mapping_func(clarity_values)
X[:, 3] = clarity_numerical


#Encoding non-numerical ordinal data into integers: 'color' attribute
color_values = X[:,2]
color_mapping = {
    'D': 0,
    'E': 1,
    'F': 2,
    'G': 3,
    'H': 4,
    'I': 5,
    'J': 6,
}
mapping_func = np.vectorize(lambda x: color_mapping[x])
color_numerical = mapping_func(color_values)
X[:, 2] = color_numerical

#Dimension of X matrix
N,M = X.shape

#Assigning float type to all matrix entries
X = X.astype(float)

#perform singular valio docompoition and create varience plot
X_mean = np.mean(X, axis=0)

X_std = np.std(X, axis=0)

Y = (X - X_mean)/X_std

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()

plt.xticks(range(1, len(rho) + 1), [i for i in range(1, len(rho) + 1)])
plt.savefig('varience_explained.png')
plt.show()

#output rho(varience explained)
sum(rho[0:2])

#data projection on pc1 and pc2
V = Vh.T
Z = Y @ V
i = 0
j = 1
f = figure()
title("Diamonds: PCA")
for c in range(C):
    class_mask = y == c
    plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
legend(classNames)
xlabel("PC{0}".format(i + 1))
ylabel("PC{0}".format(j + 1))
plt.savefig('pca1_2.png')
show()

#Vector coefficient plotted for PC1-PC5
pcs = [0, 1, 2,3,4]
legendStrs = ["PC" + str(e + 1) for e in pcs]
c = ["r", "g", "b"]
bw = 0.1
r = np.arange(1, M + 1)
plt.figure(figsize=(10, 6))
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, attributeNames, rotation='vertical')
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("Diamonds: PCA Component Coefficients")
plt.savefig('PCAComponentCoef.png')
plt.show()

#eigenvector matrix for PC1-PC5
V_matrix = np.round(V[:, :5], decimals=2)

#S matrix for PC1-PC5
S_matrix = np.round(S[:5], decimals=2)


#plots of price against each of PC1-PC5
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
PC_values = [np.dot(Y, V[:, i]) for i in range(5)]  # Compute PC values for all 5 principal components
y_values=data['price']
for i in range(2):
    for j in range(3):
        idx = i * 3 + j
        if idx < 5:
            axs[i, j].scatter(PC_values[idx], y_values, color='darkred')
            axs[i, j].set_xlabel('PC{}'.format(idx + 1))
            axs[i, j].set_title('Scatter Plot of PC{} vs price'.format(idx + 1))
            axs[i, j].grid(True)
        else:
            axs[i, j].axis('off')

plt.tight_layout()
plt.savefig('PCAsvsY.png')
plt.show()
