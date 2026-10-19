import joblib, pandas as pd, streamlit as st
st.set_page_config(page_title="Dynamic Pricing Engine",page_icon="🛒")
st.title("🛒 Dynamic Pricing Engine")
st.caption("XGBoost-based price recommendation prototype")
bundle=joblib.load("artifacts/pricing_model.joblib")
model,columns=bundle["model"],bundle["columns"]

base_price=st.number_input("Base price",20.0,10000.0,500.0)
demand=st.number_input("Current demand",0.0,1000.0,80.0)
inventory=st.number_input("Inventory units",0,10000,100)
competitor_price=st.number_input("Competitor price",20.0,10000.0,520.0)
season=st.selectbox("Season",["normal","festival","holiday"])
segment=st.selectbox("Customer segment",["budget","regular","premium"])

if st.button("Recommend price",type="primary"):
    row=pd.DataFrame([{"product_id":1,"base_price":base_price,"demand":demand,
                       "inventory":inventory,"competitor_price":competitor_price,
                       "season":season,"customer_segment":segment}])
    row=pd.get_dummies(row)
    row=row.reindex(columns=columns,fill_value=0)
    price=float(model.predict(row)[0])
    st.metric("Recommended price",f"₹{price:,.2f}")
    delta=(price-base_price)/base_price*100
    st.write(f"Suggested change from base price: **{delta:+.1f}%**")
    st.caption("Prototype recommendation only; real pricing requires business constraints and live validation.")
