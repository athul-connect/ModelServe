import pandas as pd 

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")


df["TotalCharges"]=pd. to_numeric(
    df["TotalCharges"].replace(r"^\s*$", pd.NA,regex=True),errors="coerce"
)

print("Missing values after conversion")
print(df.isnull().sum())

df = df.dropna().copy()

df=df.drop(columns=["customerID"])

print("cleaned dataset  shape")
print(df.shape)

print("Remaining missing values:")
print(df.isnull().sum().sum())

print(df.dtypes)