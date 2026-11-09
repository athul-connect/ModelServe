import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df["TotalCharges"]=pd.to_numeric(
    df["TotalCharges"].replace(r"^\s*$",pd.NA,regex=True), errors="coerce"
)

df = df.dropna().copy()

df = df.drop(columns=["customerID"])

x = df.drop(columns=["Churn"])
y = df["Churn"]

print("Features shape:",x.shape)
print("Target shape:",y.shape)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42,stratify=y)
print("Training Features:",x_train.shape)
print("Testing Features:",x_test.shape)

print("Training target distribution:")
print(y_train.value_counts(normalize=True))

print("Testing target distribution:")
print(y_test.value_counts(normalize=True))