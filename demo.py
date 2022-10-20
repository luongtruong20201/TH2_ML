from pyexpat import model
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, linear_model, tree

data = pd.read_csv('./data.csv', sep=';').to_numpy()

train, test = model_selection.train_test_split(
    data,
    test_size=0.3,
    random_state=None
)

X_train, y_train, X_test, y_test = train[:, :-
                                         1], train[:, -1:], test[:, :-1], test[:, -1:]

pla = linear_model.Perceptron()
pla.fit(X_train, y_train.ravel())
y_pred = pla.predict(X_test)
print("PLA: Ti le du doan chinh xac", end='\t')
print(pla.score(X_test, y_test))

#CART

cart = tree.DecisionTreeClassifier(criterion='gini')
cart.fit(X_train, y_train)

y_pred = cart.predict(X_test)
print("Cart: Ti le du doan chinh xac", end='\t')
print(cart.score(X_test, y_test))

# ID3
id3 = tree.DecisionTreeClassifier(criterion='entropy')
id3.fit(X_train, y_train)

y_pred = id3.predict(X_test)
print("ID3 Ti le du doan chinh xac: ", end='\t')
print(id3.score(X_test, y_test))