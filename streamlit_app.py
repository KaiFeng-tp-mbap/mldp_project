import joblib
import streamlit as st
import numpy as np
import pandas as pd

import base64

def get_number(value, field_name, cast_type=float,min_value = None):
    try: 
        num = cast_type(value)
        if min_value is not None and num < min_value:
            st.error(f"{field_name} must be >={min_value}")
            return None
        return num
    except ValueError:
        st.error(f"{field_name} must be a valid number")
        return None
    
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
bg = get_base64_of_bin_file("marketing_background.png")
## Load trained model
model = joblib.load("ifood_df_logr_adjusted_threshold_model.joblib")['model']

## Streamlit app
st.title("Marketing Prediction")
st.text("Response:")
st.text("1 if the customer accepted the offer in the last campaign \n 0 if the customer never accepted the offer in the last campaign.")
## Define the input options
Kidhome =  0
Teenhome = 0
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
Kidhome_selected = st.text_input("Enter Number of small children in household (Kidhome)", Kidhome)
Teenhome_selected = st.text_input("Enter Number of of teenagers in household (Teenhome)", Teenhome)
MntWines_selected = st.text_input("Enter Amount spent on wine in the last 2 years (MntWines)", MntWines)
Recency_selected = st.text_input("Enter number of days since last purchase (Recency) ", Recency)
MntFruits_selected= st.text_input("Enter Amount spent on Fruits in the last 2 years (MntFruits)", MntFruits)
MntMeatProducts_selected = st.text_input("Enter Amount spent on Meat in the last 2 years (MntMeatProducts)", MntMeatProducts)

MntFishProducts_selected = st.text_input("Enter Amount spent on Fish in the last 2 years (MntFishProducts)", MntFishProducts)
MntSweetProducts_selected = st.text_input("Enter Amount spent on sweets in the last 2 years (MntSweetProducts)", MntSweetProducts)
MntGoldProds_selected = st.text_input("Enter Amount spent on Gold in the last 2 years (MntGoldProds)", MntGoldProds)
NumDealsPurchases_selected= st.text_input("Enter Number of purchases made with discount (NumDealsPurchases)", NumDealsPurchases)
NumWebPurchases_selected = st.text_input("Enter Number of purchases made through company's website (NumWebPurchases)", NumWebPurchases)
NumCatalogPurchases_selected = st.text_input("Enter Number of purchases using catalog (NumCatalogPurchases)", NumCatalogPurchases)
NumStorePurchases_selected = st.text_input("Enter Number of purchases directly in store (NumStorePurchases)", NumStorePurchases)
NumWebVisitsMonth_selected = st.text_input("Enter Number of visits to the company's website in the last month (NumWebVisitsMonth)", NumWebVisitsMonth)
Complain_selected = st.selectbox("Select Complain if 1 customer Complained in the last 2 years", Complain)
Income_selected = st.text_input("Enter amount of yealy household Income", Income)
Z_CostContact_selected = st.text_input("Enter Z_CostContact", Z_CostContact)
Z_Revenue_selected = st.text_input("Enter Z_Revenue", Z_Revenue)
Age_selected = st.text_input("Enter Age", Age)
Customer_Days_selected = st.text_input("Enter date of customor's enrollment with the comapny (Customer_Days)", Customer_Days)

marital_Married_selected = st.selectbox("Select marital status (marital_Married): 1 if you are Married, 0 otherwise", marital_Married)
marital_Single_selected = st.selectbox("Select marital status (marital_Single): 1 if you are Single, 0 otherwise", marital_Single)
marital_Together_selected =  st.selectbox("Select marital status (marital_Together): 1 if you are in a relationship, 0 otherwise", marital_Together)
marital_Widow_selected =  st.selectbox("Select marital status (marital_Widow): 1 if you are Divorced, 0 otherwise", marital_Widow)
marital_Divorced_selected = st.selectbox("Select marital status (marital_Divorced): 1 if you are a widow/widower, 0 otherwise", marital_Divorced)

education_Basic_selected =  st.selectbox("Select education level (education_Basic): 1 if you have a Basic education, 0 otherwise", education_Basic)
education_Graduation_selected =  st.selectbox("Select education level (education_Graduation): 1 if you have a bachelor degree, 0 otherwise", education_Graduation)
education_Master_selected =  st.selectbox("Select education level (education_Master): 1 if you have a masters degree, 0 otherwise", education_Master)
education_PhD_selected = st.selectbox("Select education level (education_PhD): 1 if you have a PhD, 0 otherwise", education_PhD)
education_2n_Cycle_selected = st.selectbox("Select education level (education_2n_Cycle): 1 if you have a a secondary education, 0 otherwise", education_2n_Cycle)


MntTotal_selected = st.text_input("Enter Amount spent on Total products in the last 2 years (MntTotal)", MntTotal)
MntRegularProds_selected =  st.text_input("Enter Amount spent on Regular Productss in the last 2 years (MntRegularProds)", MntRegularProds)



## Predict button
if st.button("Predict marketing Response"):
    Kidhome= get_number(Kidhome_selected,"Kidhome",int,0)
    Teenhome= get_number(Teenhome_selected,"Teenhome",int,0)
    MntFruits= get_number(MntFruits_selected,"MntFruits",int,0)
    MntWines= get_number(MntWines_selected,"MntWines",int,0)
    Recency= get_number(Recency_selected,"Recency",int,0)
    MntMeatProducts= get_number(MntMeatProducts_selected,"MntMeatProducts",int,0)

    MntFishProducts= get_number(MntFishProducts_selected,"MntFishProducts",int,0)
    MntSweetProducts= get_number(MntSweetProducts_selected,"MntSweetProducts",int,0)
    MntGoldProds= get_number(MntGoldProds_selected,"MntGoldProds",int,0)
    NumDealsPurchases= get_number(NumDealsPurchases_selected,"NumDealsPurchases",int,0)
    NumWebPurchases= get_number(NumWebPurchases_selected,"NumWebPurchases",int,0)
    NumCatalogPurchases= get_number(NumCatalogPurchases_selected,"NumCatalogPurchases",int,0)
    NumStorePurchases= get_number(NumStorePurchases_selected,"NumStorePurchases",int,0)
    NumWebVisitsMonth= get_number(NumWebVisitsMonth_selected,"NumWebVisitsMonth",int,0)
    Income= get_number(Income_selected,"Income",int,0)
    Z_CostContact= get_number(Z_CostContact_selected,"Z_CostContact",int)
    Z_Revenue= get_number(Z_Revenue_selected,"Z_Revenue",int)
    Age= get_number(Age_selected,"Age",int,0)
    Customer_Days= get_number(Customer_Days_selected,"Customer_Days",int,0)

    MntTotal= get_number(MntTotal_selected,"MntTotal",int,0)
    MntRegularProds= get_number(MntRegularProds_selected,"MntRegularProds",int,0)

    if None in [
        Kidhome,Teenhome,MntFruits,MntWines,Recency,MntMeatProducts,MntFishProducts,MntSweetProducts,
        MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,
        Income,Z_CostContact,Z_Revenue,Age,Customer_Days,MntTotal,MntRegularProds
    ]:
        st.stop()
    ## Create dict for input features
    input_data = {
        'Kidhome': Kidhome,
        'Teenhome': Teenhome,
        'MntFruits': MntFruits,
        "MntWines": MntWines,
        "Recency":Recency,
        "MntMeatProducts": MntMeatProducts,

        "MntFishProducts": MntFishProducts,
        "MntSweetProducts": MntSweetProducts,
        "MntGoldProds": MntGoldProds,
        "NumDealsPurchases": NumDealsPurchases,
        "NumWebPurchases": NumWebPurchases,
        "NumCatalogPurchases": NumCatalogPurchases,
        "NumStorePurchases": NumStorePurchases,
        "NumWebVisitsMonth": NumWebVisitsMonth,
        "Complain": Complain_selected,
        "Income": Income,
        "Z_CostContact": Z_CostContact,
        "Z_Revenue": Z_Revenue,
        "Age": Age,
        "Customer_Days": Customer_Days,

        "marital_Married": marital_Married_selected,
        "marital_Single": marital_Single_selected,
        "marital_Together": marital_Together_selected,
        "marital_Widow": marital_Widow_selected,
        "education_Basic": education_Basic_selected,
        "education_Graduation": education_Graduation_selected,
        "education_Master": education_Master_selected,
        "education_PhD": education_PhD_selected,
        "MntTotal": MntTotal,
        "MntRegularProds": MntRegularProds,

        "education_2n_Cycle": education_2n_Cycle_selected,
        "marital_Divorced": marital_Divorced_selected,

    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame([{
        'Kidhome': Kidhome,
        'Teenhome': Teenhome,
        'MntFruits': MntFruits,
        'MntWines': MntWines,
        'Recency':Recency,
        'MntMeatProducts': MntMeatProducts,

        "MntFishProducts": MntFishProducts,
        "MntSweetProducts": MntSweetProducts,
        "MntGoldProds": MntGoldProds,
        "NumDealsPurchases": NumDealsPurchases,
        "NumWebPurchases": NumWebPurchases,
        "NumCatalogPurchases": NumCatalogPurchases,
        "NumStorePurchases": NumStorePurchases,
        "NumWebVisitsMonth": NumWebVisitsMonth,
        "Complain": Complain_selected,
        "Z_CostContact": Z_CostContact,
        "Z_Revenue": Z_Revenue,
        "Age": Age,
        "Customer_Days": Customer_Days,

        "marital_Married": marital_Married_selected,
        "marital_Single": marital_Single_selected,
        "marital_Together": marital_Together_selected,
        "marital_Widow": marital_Widow_selected,
        "education_Basic": education_Basic_selected,
        "education_Graduation": education_Graduation_selected,
        "education_Master": education_Master_selected,
        "education_PhD": education_PhD_selected,
        "MntTotal": MntTotal,
        "MntRegularProds": MntRegularProds,

        "education_2n_Cycle": education_2n_Cycle_selected,
        "marital_Divorced": marital_Divorced_selected,
        "Income": Income
    }])

    ## One-hot encoding
    df_input = pd.get_dummies(df_input, 
                              columns = ['Kidhome', 'Teenhome', 'Complain','Z_CostContact','Z_Revenue','marital_Married','marital_Single',
                                         'marital_Together','marital_Widow','education_Basic','education_Graduation','education_Master','education_PhD','education_2n_Cycle','marital_Divorced']
                             )


    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)



    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Marketing campaign Response: {y_unseen_pred}")

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