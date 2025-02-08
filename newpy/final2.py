import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def preprocess_data(data):
    X = data.iloc[:, :-10]  # entekhab features haa

    y = data.iloc[:, -10:]  # entekhab lable

    if X.isna().any().any():
        print("Warning: NaN values found in feature columns. Filling NaN values with the mean.")
        X = X.fillna(X.mean()) 

    if y.isna().any().any():
        print("Warning: NaN values found in target columns. Filling NaN values with 0.")
        y = y.fillna(0) 

    for drug in y.columns:
        print(f"Distribution of {drug}:")
        print(y[drug].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
# ghesmat train bar asas 2 model
def train_models(X_train, y_train):
    models = {}
    for drug in y_train.columns:
        if y_train[drug].nunique() == 1:
            continue

        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
        svm_model.fit(X_train, y_train[drug])

        lr_model = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced')
        lr_model.fit(X_train, y_train[drug])

        models[drug] = {'SVM': svm_model, 'LogisticRegression': lr_model}

    return models
#pishbini 
def predict_drug_prescription(models, user_input, scaler):
    user_input_scaled = scaler.transform([user_input])

    predictions = {}
    for drug, model in models.items():
        svm_prediction = model['SVM'].predict(user_input_scaled)
        lr_prediction = model['LogisticRegression'].predict(user_input_scaled)
        predictions[drug] = {'SVM': svm_prediction[0], 'LogisticRegression': lr_prediction[0]}

    return predictions
#summry az model dar har class
def compare_models(models, X_test, y_test):
    for drug in y_test.columns:
        if drug not in models:
            continue  

        svm_predictions = models[drug]['SVM'].predict(X_test)
        lr_predictions = models[drug]['LogisticRegression'].predict(X_test)

        print(f"Drug {drug}:")
        print("  SVM Classification Report:")
        print(classification_report(y_test[drug], svm_predictions))
        print("  Logistic Regression Classification Report:")
        print(classification_report(y_test[drug], lr_predictions))


def main():
    file_path = "feshar2.xlsx"  
    data = pd.read_excel(file_path , sheet_name='Sheet1')
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')


    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    models = train_models(X_train, y_train)
    compare_models(models, X_test, y_test)    
    user_input = []  
    print("Enter the following features:")
    features = data.columns[:-10] 
    for feature in features:
        value = float(input(f"{feature}: "))
        user_input.append(value)

    predictions = predict_drug_prescription(models, user_input, scaler)

    print("\nDrug Prescription Predictions:")
    for drug, pred in predictions.items():
        print(f"Drug {drug}:")
        print(f"  SVM Prediction: {'Use' if pred['SVM'] == 1 else 'Do not use'}")
        print(f"  Logistic Regression Prediction: {'Use' if pred['LogisticRegression'] == 1 else 'Do not use'}")

if __name__ == "__main__":
    main()
