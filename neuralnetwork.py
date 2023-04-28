import streamlit as st
from streamlit.components.v1 import components
import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns             # visualizations
import matplotlib.pyplot as plt   # visualizations
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Nadam

from keras.models import model_from_json

from datetime import date


df = pd.read_csv('Dataset/cardio_train_cleaned_1.csv')

class NeuralNetwork:

    def hello(self):
        f_date = st.date_input('Enter your DOB')
        l_date = date.today()
        delta = l_date - f_date
        agedays=delta.days

        gender = st.number_input('Enter gender: 0 for woman; 1 for man', min_value=0, max_value=1)
        height = st.number_input('Enter your height (in cms)', min_value=0, max_value=300)
        weight = st.number_input('Enter your weight (in kgs)', min_value=0, max_value=500)
        systolicbloodpressure= st.number_input('Enter your systolic blood pressure', max_value=250)
        diastolicbloodpressure = st.number_input('Enter your diastolic blood pressure', min_value=30, max_value=100)
        cholesterol = st.number_input('Enter cholesterol level 1: normal, 2: above normal, 3: well above normal', max_value=3, min_value=1)
        gluc = st.number_input('Enter glucose level 1: normal, 2: above normal, 3: well above normal', min_value=1, max_value=3)
        smoke = st.number_input('Do you smoke? 1 if you smoke, 0 if not', min_value=0, max_value=1)
        alco = st.number_input('Do you drink alcohol? 1 if you smoke, 0 if not', min_value=0, max_value=1)
        active = st.number_input('Do you exercise regularly? 1 if you do, 0 if not', min_value=0, max_value=1)


        # Converting all of these inputs into proper scale to feed into the NN

        agedayscale=(agedays-df["age"].min())/(df["age"].max()-df["age"].min())
        heightscale=(height-df["height"].min())/(df["height"].max()-df["height"].min())
        weightscale=(weight-df["weight"].min())/(df["weight"].max()-df["weight"].min())
        sbpscale=(systolicbloodpressure-df["ap_hi"].min())/(df["ap_hi"].max()-df["ap_hi"].min())
        dbpscale=(diastolicbloodpressure-df["ap_lo"].min())/(df["ap_lo"].max()-df["ap_lo"].min())
        cholesterolscale=(cholesterol-df["cholesterol"].min())/(df["cholesterol"].max()-df["cholesterol"].min())
        glucscale=(gluc-df["gluc"].min())/(df["gluc"].max()-df["gluc"].min())

        single=np.array([agedayscale, gender, heightscale, weightscale, sbpscale, dbpscale, cholesterolscale, glucscale, smoke, alco, active ])
        singledf=pd.DataFrame(single)
        final=singledf.transpose()
        # st.dataframe(final)

        # loading the model

        with open('Predictions/model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        # loading thr weights
        loaded_model.load_weights("Predictions/weights_final.hdf5")






        # st.write(height) 

        