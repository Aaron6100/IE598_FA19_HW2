#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
df = pd.read_csv("data.csv", sep = ',')
#import the data from local


# In[49]:


# Create arrays for the features and the response variable
import numpy as np
X = df.iloc[1:, 6:8]#test a couple of times finding the graph of this one looks neat
y = df.iloc[1:,-1]
print(X.shape,y.shape)
#get the specific data from the list excluding those useless columns and rows
#"shape" to know how many data specifically we have
#X is all of the features and y is all of the targets (what kind of flowers for the iris set but this one is different)


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# spliting the datasets into train(30%) and test groups


# In[51]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#fit the StandardScaler into our X_train sets


# In[52]:


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 
# Import accuracy_score to test the accuracy
from sklearn.metrics import accuracy_score
#try k=1 through k=25 and record testing accuracy
k_range=range(1,26)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
    
# Generate plot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# In[53]:


scores
np.max(scores)
best_try = scores.index(np.max(scores)) + 1
#The reason why we are adding one is because we 
#are counting from 0


# In[54]:


#this whole giant thing is for defining plot_decision_regions

def plot_decision_regions(X, y, classifier, test_idx = None,
                          resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
       # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                              np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                       alpha=0.8, c=colors[idx],
                       marker=markers[idx], label=cl,
                       edgecolor='black')
       # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                       c='', edgecolor='black', alpha=1.0,
                       linewidth=1, marker='o',
                       s=100, label='test set')


# In[55]:


# we gotta "std" all the X but not y then bascially plot everything
knn_endgame = KNeighborsClassifier(n_neighbors=best_try)
X_combined = np.vstack((X_train_std, X_test_std)) 
X_combined_std = sc.transform(X_combined)
y_combined = np.hstack((y_train, y_test))
knn_endgame.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,classifier=knn_endgame, test_idx=range(105,150))
plt.xlabel('near_minus_next [standardized]')
plt.ylabel('ctd_last_first [standardized]')
plt.legend(loc='upper left')
plt.show()


#------------ End of the KNN part -------------#


# In[56]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train, y_train)
new_X_combined = np.vstack((X_train, X_test))
new_y_combined = np.hstack((y_train, y_test))
plot_decision_regions(new_X_combined, new_y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('near_minus_next [pg]')
plt.ylabel('std_last_first [many]')
plt.legend(loc='upper left')
plt.show()


#------------ End of the DecisionTree part -------------#


# In[57]:


print("My name is {Qianyi Liu}")
print("My NetID is: {qianyil2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




