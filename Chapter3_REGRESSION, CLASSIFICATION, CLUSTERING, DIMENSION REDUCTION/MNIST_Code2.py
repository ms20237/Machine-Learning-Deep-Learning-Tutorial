import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE


import tensorflow 
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras import Sequential


for dirname, _, filenames in os.walk('Input\t10k-images.idx3-ubyte'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

print(X_train.shape)
print(X_test.shape)

print(y_train)

plt.imshow(X_train[2])
plt.show()

# Normalization of Data(Pixel values are often stored as integers in the range of 0 to 255 easy for PCA to model)
X_train = X_train / 255
X_test = X_test / 255

for i in range(len(X_train))[:2]:
    # Apply PCA
    pca = PCA(n_components=20)  # Reduce to 2 components for visualization
    X_pca = pca.fit_transform(X_train[i])


    X_train_pca = pca.fit_transform(X_train[i])
    X_test_pca = pca.transform(X_test[i])

    # Check the explained variance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Plot the explained variance ratio
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot use PCA')
    plt.show()
    
    
    
pred_value=4
act_value=9
confusions=int(confusion_matrix[act_value][pred_value])
fig, axes = plt.subplots(ncols=confusions, sharex=True, sharey=True, figsize=(20, 10))
confusions=0
for i in range(len(y_test)):
    if y_test[i]==pred_value:
        if y_test[i]==act_value:
            #axes[i].set_title(act_value, " ", pred_value)
            axes[confusions].imshow(X_test[i], cmap='gray')
            axes[confusions].get_xaxis().set_visible(False)
            axes[confusions].get_yaxis().set_visible(False)
            confusions += 1

print("Actual value:", act_value, ", Predicted value:", pred_value)
plt.show()
print("Total Confusions:", confusions)







# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train[0])
# X_test_scaled = scaler.transform(X_test[0])

# print(X_train_scaled.shape)

# n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
# tsne = TSNE(n_components=n_components, random_state=42)
# X_train_tsne = tsne.fit_transform(X_train_scaled)

# plt.figure(figsize=(10, 6))
# plt.scatter(X_train_tsne[:, 0], y_train, cmap='viridis')
# plt.colorbar()
# plt.title('t-SNE Visualization')
# plt.xlabel('t-SNE Component')
# plt.ylabel('Target')
# plt.show()


# def fit_random_forest_classifier(X, y, print_output=True):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)            
                                                        
#     clf = RandomForestClassifier(n_estimators=100, max_depth=None)

#     clf.fit(X_train, y_train)

#     y_preds = clf.predict(X_test)

#     acc = accuracy_score(y_test, y_preds)
    
#     if print_output == True:
#         mat = confusion_matrix(y_test, y_preds)
#         sns.heatmap(mat, annot=True, cmap='bwr', linewidths=.5)

#         print('Input Shape: {}'.format(X_train.shape))
#         print('Accuracy: {:2.2%}\n'.format(acc))
#         print(mat)
    
#     return acc

# fit_random_forest_classifier(X_train[0], y_train[0]);


# def do_pca(n_components, data):
#     X = StandardScaler().fit_transform(data)
#     pca = PCA(n_components)
#     X_pca = pca.fit_transform(X)
#     return pca, X_pca

# pca, X_pca = do_pca(2, X_train[0])
# fit_random_forest_classifier(X_pca, y_train[0]);
