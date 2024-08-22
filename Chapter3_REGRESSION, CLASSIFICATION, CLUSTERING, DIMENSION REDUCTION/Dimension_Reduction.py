import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from matplotlib.colors import ListedColormap


# Reading the data from the .dat file
file_path = 'flutter.dat' 
with open(file_path, 'r') as file: 
    content = file.read()

# Create a StringIO object to simulate a file for pandas
data = StringIO(content)

# Reading the data into a DataFrame
df = pd.read_csv(data, delim_whitespace=True, header=None)



# Check the first few rows to understand the structure
print(df.head())
print(df.shape)  # Print the shape to check the number of columns
print(df.keys())


# Extracting input (X) and output (y) values
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column


# Plotting the data
plt.scatter(X, y)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Scatter plot of Input vs Output')
plt.show()



# Correlation heatmaps
correlation_matrix = df.corr(numeric_only = True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA
n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check the explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot the explained variance ratio
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot use PCA')
plt.show()




# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_train_scaled.shape)

n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
tsne = TSNE(n_components=n_components, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_tsne[:, 0], y_train, cmap='viridis')
plt.colorbar()
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component')
plt.ylabel('Target')
plt.show()


# Perform LDA
lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)
X_lda = lda.transform(X_test)  # Transform the data
print("Explained Variance Ratio:", lda.explained_variance_ratio_)

# Assuming you have already performed LDA and have the transformed data in X_lda
# Plot the transformed data
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.title('LDA Transformed Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()
