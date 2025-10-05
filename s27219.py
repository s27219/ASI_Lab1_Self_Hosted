# train.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Generowanie prostego zbioru danych
def generate_data():
    center1 = [0, 0]
    center2 = [5, 5]

    n1 = np.random.randint(50, 101)
    n2 = np.random.randint(50, 101)

    cloud1 = np.random.randn(n1, 2) + center1
    cloud2 = np.random.randn(n2, 2) + center2

    X = np.vstack((cloud1, cloud2))
    y = np.hstack((np.zeros(n1), np.ones(n2)))

    return X, y


# Trenowanie prostego modelu regresji logistycznej
def train_model():

    X, y = generate_data()
   
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Trenowanie modelu
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)
    
    # Wyliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)
    
    # Zapis wyniku
    with open("accuracy.txt", "w") as f:
        f.write(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_model()
