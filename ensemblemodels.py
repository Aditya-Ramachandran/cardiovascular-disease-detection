import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

class EnsembleModels():

    def __init__(self):
        self.__sex = st.selectbox('Enter your sex', options=('M', 'F'))
        self.__age = st.number_input('Enter your age', min_value=1, max_value=110)
        self.__pain_type = st.selectbox('Enter your chest pain type', options=('ASY','NAP','ATA','TA'))
        self.__cholesterol = st.number_input('Enter your cholesterol', max_value=400)
        self.__FastingBS = st.selectbox('Enter your fasting blood sugar (0 -> Normal; 1-> High)', options=(0,1))
        self.__maxhr = st.number_input('Enter your maximum heart rate', max_value=250)
        self.__exerangina = st.selectbox('Do you have exercise angina (N -> No; Y -> Yes)', options=('N', 'Y'))
        self.__oldpeak = st.number_input('Enter old peak range')
        self.__st_slope = st.selectbox('Enter ST Slope codition', options=('Up','Down','Flat'))
    

    def __preprocess(self):

        if self.__sex == 'M':
            self.__sex = 0
        else:
            self.__sex = 1
    
        if self.__pain_type == 'NAP':
            self.__pain_type = 2
        elif self.__pain_type == 'ATA':
            self.__pain_type = 1
        elif self.__pain_type == 'ASY':
            self.__pain_type = 0
        elif self.__pain_type == 'TA':
            self.__pain_type = 3
            
        if self.__exerangina == 'N':
            self.__exerangina = 0
        else:
            self.__exerangina = 1

        if self.__st_slope == 'Up':
            self.__st_slope = 2
        elif self.__st_slope == 'Flat':
            self.__st_slope = 1
        elif self.__st_slope == 'Down':
            self.__st_slope = 0
        
        data = {
            'age': self.__age,
            'sex': self.__sex,
            'pain_type': self.__pain_type,
            'cholesterol': self.__cholesterol,
            'FastingBS': self.__FastingBS,
            'maxhr': self.__maxhr,
            'exerangina': self.__exerangina,
            'oldpeak': self.__oldpeak,
            'st_slope': self.__st_slope
        }

        df1 = pd.DataFrame(data, index=[0])
        return df1

        
    def predict(self):
        btn = st.button('Predict')
        if btn:
            model_dt = pickle.load(open('Predictions/Models/dt_model.pkl', 'rb'))
            model_knn = pickle.load(open('Predictions/Models/knn_model.pkl', 'rb'))
            model_lr = pickle.load(open('Predictions/Models/lr_model.pkl', 'rb'))
            model_rf = pickle.load(open('Predictions/Models/rf_model.pkl', 'rb'))
            model_svm = pickle.load(open('Predictions/Models/svc_model.pkl', 'rb'))

            df = self.__preprocess()
            query = df.values

            models = {
            "Decision Tree": model_dt,
            "K-Nearest Neighbors": model_knn,
            "Logistic Regression": model_lr,
            "Random Forest": model_rf,
            "Support Vector Machine": model_svm
        }

            predictions = []

            for model_name, model in models.items():
                prediction = model.predict(query)
                predictions.append(prediction)
                result = "you DO NOT have heart disease" if prediction == 0 else "you HAVE a chance of heart disease"
                st.write(f"According to {model_name}: {result}\n")

            flat_predictions = np.ravel(predictions)
            majority_vote = np.bincount(flat_predictions).argmax()
            final_result = "you DO NOT have heart disease" if majority_vote == 0 else "you HAVE a chance of heart disease"
            st.write(f"\nFinal Result (Majority Vote): {final_result}")