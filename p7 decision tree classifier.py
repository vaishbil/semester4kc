from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report 
# Load dataset 
data = load_wine() 
X, y = data.data, data.target 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
# Scale data 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
# k-NN 
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 
print("k-NN Classification Report:") 
print(classification_report(y_test, knn.predict(X_test))) 
# Decision Tree 
tree = DecisionTreeClassifier() 
tree.fit(X_train, y_train) 
print("Decision Tree Classification Report:") 
print(classification_report(y_test, tree.predict(X_test))) 