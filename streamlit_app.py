import joblib
import streamlit as st
import numpy as np
import pandas as pd

import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
bg = get_base64_of_bin_file("marketing_background.png")
## Load trained model
model = joblib.load("ifood_df_rf_adjusted_threshold_model.joblib")['model']

## Streamlit app
st.title("Marketing Prediction")

## Define the input options
Kidhome = [2, 1, 0]
Teenhome = [2, 1, 0]
MntWines = 0 
Recency = 0
MntFruits=  0
MntMeatProducts = 0
MntFishProducts = 0
MntSweetProducts = 0
MntGoldProds = 0
NumDealsPurchases= 0
NumWebPurchases = 0
NumCatalogPurchases = 0 
NumStorePurchases = 0
NumWebVisitsMonth = 0
Complain = [0,1]
Income = 0
Z_CostContact = 0
Z_Revenue = 0 
Response = [0,1]
Age = 0
Customer_Days = 0 
marital_Divorced = [0,1]
marital_Married = [0,1]
marital_Single = [0,1]
marital_Together = [0,1]
marital_Widow = [0,1]
education_Basic = [0,1]
education_Graduation = [0,1]
education_Master = [0,1]
education_PhD = [0,1]
MntTotal = 0 
MntRegularProds = 0
education_2n_Cycle = [0,1]
## User inputs
Kidhome_selected = st.selectbox("Select Kidhome", Kidhome)
Teenhome_selected = st.selectbox("Select Teenhome", Teenhome)
MntWines_selected = st.text_input("Enter MntWines", MntWines)
Recency_selected = st.text_input("Enter Recency", Recency)
MntFruits_selected= st.text_input("Enter MntFruits", MntFruits)
MntMeatProducts_selected = st.text_input("Enter MntMeatProducts", MntMeatProducts)

MntFishProducts_selected = st.text_input("Enter MntFishProducts", MntFishProducts)
MntSweetProducts_selected = st.text_input("Enter MntSweetProducts", MntSweetProducts)
MntGoldProds_selected = st.text_input("Enter MntGoldProds", MntGoldProds)
NumDealsPurchases_selected= st.text_input("Enter NumDealsPurchases", NumDealsPurchases)
NumWebPurchases_selected = st.text_input("Enter NumWebPurchases", NumWebPurchases)
NumCatalogPurchases_selected = st.text_input("Enter NumCatalogPurchases", NumCatalogPurchases)
NumStorePurchases_selected = st.text_input("Enter NumStorePurchases", NumStorePurchases)
NumWebVisitsMonth_selected = st.text_input("Enter NumWebVisitsMonth", NumWebVisitsMonth)
Complain_selected = st.selectbox("Select Complain", Complain)
Income_selected = st.text_input("Enter Income", Income)
Z_CostContact_selected = st.text_input("Enter Z_CostContact", Z_CostContact)
Z_Revenue_selected = st.text_input("Enter Z_Revenue", Z_Revenue)
Response_selected = st.selectbox("Select Response", Response)
Age_selected = st.text_input("Enter Age", Age)
Customer_Days_selected = st.text_input("Enter Customer_Days", Customer_Days)

marital_Married_selected = st.selectbox("Select marital_Married", marital_Married)
marital_Single_selected = st.selectbox("Select marital_Single", marital_Single)
marital_Together_selected =  st.selectbox("Select marital_Together", marital_Together)
marital_Widow_selected =  st.selectbox("Select marital_Widow", marital_Widow)
education_Basic_selected =  st.selectbox("Select education_Basic", education_Basic)
education_Graduation_selected =  st.selectbox("Select education_Graduation", education_Graduation)
education_Master_selected =  st.selectbox("Select education_Master", education_Master)
education_PhD_selected = st.selectbox("Select education_PhD", education_PhD)
MntTotal_selected = st.text_input("Enter MntTotal", MntTotal)
MntRegularProds_selected =  st.text_input("Enter MntRegularProds", MntRegularProds)

education_2n_Cycle_selected = st.selectbox("Select education_2n_Cycle", education_2n_Cycle)
marital_Divorced_selected = st.selectbox("Select marital_Divorced", marital_Divorced)

## Predict button
if st.button("Predict marketing Response"):

    ## Create dict for input features
    input_data = {
        'Kidhome': Kidhome_selected,
        'Teenhome': Teenhome_selected,
        'MntFruits': MntFruits_selected,
        "MntWines": MntWines_selected,
        "Recency":Recency_selected,
        "MntMeatProducts": MntMeatProducts_selected,

        "MntFishProducts": MntFishProducts_selected,
        "MntSweetProducts": MntSweetProducts_selected,
        "MntGoldProds": MntGoldProds_selected,
        "NumDealsPurchases": NumDealsPurchases_selected,
        "NumWebPurchases": NumWebPurchases_selected,
        "NumCatalogPurchases": NumCatalogPurchases_selected,
        "NumStorePurchases": NumStorePurchases_selected,
        "NumWebVisitsMonth": NumWebVisitsMonth_selected,
        "Complain": Complain_selected,
        "Income": Income_selected,
        "Z_CostContact": Z_CostContact_selected,
        "Z_Revenue": Z_Revenue_selected,
        "Response": Response_selected,
        "Age": Age_selected,
        "Customer_Days": Customer_Days_selected,

        "marital_Married": marital_Married_selected,
        "marital_Single": marital_Single_selected,
        "marital_Together": marital_Together_selected,
        "marital_Widow": marital_Widow_selected,
        "education_Basic": education_Basic_selected,
        "education_Graduation": education_Graduation_selected,
        "education_Master": education_Master_selected,
        "education_PhD": education_PhD_selected,
        "MntTotal": MntTotal_selected,
        "MntRegularProds": MntRegularProds_selected,

        "education_2n_Cycle": education_2n_Cycle_selected,
        "marital_Divorced": marital_Divorced_selected,

    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'Kidhome': [Kidhome_selected],
        'Teenhome': [Teenhome_selected],
        'MntFruits': [MntFruits_selected],
        "MntWines": [MntWines_selected],
        "Recency":[Recency_selected],
        "MntMeatProducts": [MntMeatProducts_selected],

        "MntFishProducts": [MntFishProducts_selected],
        "MntSweetProducts": [MntSweetProducts_selected],
        "MntGoldProds": [MntGoldProds_selected],
        "NumDealsPurchases": [NumDealsPurchases_selected],
        "NumWebPurchases": [NumWebPurchases_selected],
        "NumCatalogPurchases": [NumCatalogPurchases_selected],
        "NumStorePurchases": [NumStorePurchases_selected],
        "NumWebVisitsMonth": [NumWebVisitsMonth_selected],
        "Complain": [Complain_selected],
        "Z_CostContact": [Z_CostContact_selected],
        "Z_Revenue": [Z_Revenue_selected],
        "Response": [Response_selected],
        "Age": [Age_selected],
        "Customer_Days": [Customer_Days_selected],

        "marital_Married": [marital_Married_selected],
        "marital_Single": [marital_Single_selected],
        "marital_Together": [marital_Together_selected],
        "marital_Widow": [marital_Widow_selected],
        "education_Basic": [education_Basic_selected],
        "education_Graduation": [education_Graduation_selected],
        "education_Master": [education_Master_selected],
        "education_PhD": [education_PhD_selected],
        "MntTotal": [MntTotal_selected],
        "MntRegularProds": [MntRegularProds_selected],

        "education_2n_Cycle": [education_2n_Cycle_selected],
        "marital_Divorced": [marital_Divorced_selected],
        "Income": [Income_selected]
    })

    ## One-hot encoding
    df_input = pd.get_dummies(df_input, 
                              columns = ['Kidhome', 'Teenhome', 'Complain','Z_CostContact','Z_Revenue','Response','marital_Married','marital_Single',
                                         'marital_Together','marital_Widow','education_Basic','education_Graduation','education_Master','education_PhD','education_2n_Cycle','marital_Divorced']
                             )


    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)



    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Marketing campaign Response: {y_unseen_pred:,.3f}")

## Page design
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{bg}");
        background-size: cover
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)