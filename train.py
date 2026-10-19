import os, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

rng=np.random.default_rng(7)
N=12000
df=pd.DataFrame({
    "product_id":rng.integers(1,101,N),
    "base_price":rng.uniform(100,2000,N),
    "demand":rng.gamma(3,20,N),
    "inventory":rng.integers(5,500,N),
    "competitor_price":rng.uniform(80,2100,N),
    "season":rng.choice(["normal","festival","holiday"],N,p=[.65,.25,.10]),
    "customer_segment":rng.choice(["budget","regular","premium"],N)
})
season_mult=df["season"].map({"normal":1.0,"festival":1.25,"holiday":1.15})
segment_mult=df["customer_segment"].map({"budget":.92,"regular":1.0,"premium":1.08})
# Simulated optimal price target.
df["optimal_price"] = (
    df["base_price"]*(0.72 + 0.003*df["demand"] + 0.0008*df["inventory"])
    + 0.30*df["competitor_price"]
) * season_mult * segment_mult
df["optimal_price"] = df["optimal_price"].clip(20,5000)

X=pd.get_dummies(df.drop(columns=["optimal_price"]), drop_first=False)
y=df["optimal_price"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model=XGBRegressor(
    n_estimators=350,max_depth=6,learning_rate=.06,
    subsample=.85,colsample_bytree=.9,objective="reg:squarederror",
    random_state=42
)
model.fit(Xtr,ytr)
pred=model.predict(Xte)
mae=mean_absolute_error(yte,pred)
rmse=mean_squared_error(yte,pred)**0.5
r2=r2_score(yte,pred)
print("MAE:",round(mae,2),"RMSE:",round(rmse,2),"R2:",round(r2,4))
os.makedirs("artifacts",exist_ok=True)
joblib.dump({"model":model,"columns":list(X.columns)}, "artifacts/pricing_model.joblib")
print("Saved artifacts/pricing_model.joblib")
