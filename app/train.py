import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocessing import create_preprocessor


df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df["TotalCharges"] =pd.to_numeric(
    df["TotalCharges"].replace(r"^\s*$",pd.NA,regex=True),errors="coerce"
)

df = df.dropna().copy()
df = df.drop(columns=["customerID"])

x = df.drop(columns=["Churn"])
y=df["Churn"].map({"No":0,"Yes":1})
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42,stratify=y)

preprocessor = create_preprocessor(x_train)

model = Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
])

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save the complete pipeline
joblib.dump(model, "models/churn_pipeline.joblib")

print("\nModel saved successfully.")