import streamlit as st
import pandas as pd
from func import FeatureSelector
from charts import *
from models import ModelRunner
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

df = None # global dataframe; if it is not uploaded, it is None.

def main():
    global df
    #Title
    st.title("Feature Selection App")
    #sidebar
    st.sidebar.subheader("File Upload")
    uploaded_file = st.sidebar.file_uploader(label="Upload your file, Only .csv or .xlsx",type=['csv','xlsx'])
    #file upload
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_excel(uploaded_file)
    #after upload
    if df is not None:
        #display dataframe
        st.write(df)
        #select target variable
        target = st.selectbox("Select Target Feature",df.columns)
        #select feature selection method
        selector = st.radio(label="Selection Method",options=["SelectKBest","RFE","SelectFromModel"])
        F = FeatureSelector(df,target)
        univariate,ref,sfm,problem = F.get_result_dictionaries()
        #chart
        if selector == "SelectKBest":
            fig = barchart(univariate["feature_names"], univariate["scores"],"Feature Scores acc to SelectKBest")
        elif selector == "RFE":
            fig = barchart(ref["feature_names"], ref["ranking"],"Ranking acc to RFE; (Lower better)")
        elif selector == "SelectFromModel":
            fig = barchart(sfm["feature_names"], sfm["scores"],"Feature Scores acc to SelectFromModel")
        st.pyplot(fig)
        #select k number of features to proceed
        k = st.number_input("Number of Feature to proceed (k): ", min_value=0, max_value= len(df.columns) - 1)
        if problem == "regression":
            model = st.selectbox("ML Method",["Linear Regression","XGBoost"])
        else:
            model = st.selectbox("ML Method",["Logistic Regression","Decision Tree"])
        #when k is determined 
        if k > 0:
            #get last X,y according to feature selection
            X,_,temp,col_types,_ = F.extract_x_y() 
            y = df[target].values.reshape(-1,1)
            #feature set
            if selector == "SelectKBest":
                X = F.univariate_feature_selection(X,y,temp,k)["X"]
            elif selector == "RFE":
                X = F.ref_feature_selection(X,y,temp,col_types,k)["X"]
            elif selector == "SelectFromModel":
                X = F.sfm_feature_selection(X,y,temp,col_types,k)["X"]
            #run models
            M = ModelRunner(model,X,y,problem)
            score = M.runner()
            #display score
            st.write("Score of Model: {}".format(score))

main()