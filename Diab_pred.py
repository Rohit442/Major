import numpy as np
import pickle
import pandas as pd
# from flasgger import Swagger
import streamlit as st


# app=Flask(__name__)
# Swagger(app)

pickle_in = open("Maj_proj_model_pickle", "rb")
classifier = pickle.load(pickle_in)


# @app.route('/')
#def welcome():
#    return "Welcome All"

# @app.route('/predict',methods=["Get"])
def Diabetes_prediction(Preg,Gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    prediction = classifier.predict([[Preg,Gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(prediction)
    return prediction


def main():
    st.title("Diabetes Predictor")
    html_temp = """
    <div style="background-color:#546beb;padding:10px">
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)    
    Preg =st.sidebar.slider("Pregnancies",0,20,6)
    Gluc = st.sidebar.slider("Glucose",0,200,140)
    BP = st.sidebar.slider("BloodPressure",0,120,80)
    SkinThickness = st.sidebar.slider("SkinThickness",0,99,32)
    Insulin = st.sidebar.slider("Insulin",0,846,127)
    BMI = st.sidebar.slider("BMI",0.0,70.0,35.0)
    DiabetesPedigreeFunction = st.sidebar.slider("DiabetesPedigreeFunction",0.078,2.42,0.62)
    Age = st.sidebar.slider("Age",21,81,41)

    
    result = ""
    if st.button("Predict"):
        result = Diabetes_prediction(int(Preg),int(Gluc),int(BP),int(SkinThickness),int(Insulin),float(BMI), float(DiabetesPedigreeFunction),int(Age))
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


#if __name__ == '__main__':
main()
