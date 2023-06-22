import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from about import About
from explore import Explore
# from neuralnetwork import NeuralNetwork
from ensemblemodels import EnsembleModels


st.set_page_config(page_title="Cardiovascular Disease Detection in Python")
# st.title('Cardiovascular Disease Detection Using Python !')
# st.subheader('Major Project')

st.set_option('deprecation.showPyplotGlobalUse', False)


df = pd.read_csv('Dataset/cardio_train_cleaned_1.csv')

########################################################################################################

# function for custom scatter plots
def custom_scatter(dataframe,x,y, xx, yy, label1, label2, x_axis,y_axis):
#     xx = random.sample(range(0, 100), 50)
#     yy = random.sample(range(0, 100), 50)
    plt.scatter(xx, yy, alpha=0.1, color='orange')
    plt.scatter(x,y, color='red', marker='x', alpha=1, label='You')
    plt.text(x+0.5,y+0.5,'You are here', fontsize='large', stretch='semi-expanded')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.axhline(df[label2].mean(), linestyle='dotted', alpha=0.5)
#     plt.text(df['age'].mean()+5,df['weight'].mean()+5,'Average',rotation=90, alpha=0.5)
    plt.axvline(df[label1].mean(), linestyle='dotted', alpha=0.5)




    # # with open("flower.png", "rb") as file:
    # btn = st.download_button(
    #         label="Download image",
    #         data=file,
    #         file_name="flower.png",
    #         mime="image/png"
    #     )

########################################################################################################

# nn_obj = NeuralNetwork()

# function for model_option
# def NeuralNetwork():
#     nn_obj.hello()


# def EnsembleModels():



st.sidebar.header('Cardiovascular Disease Detection Using Python')
option = st.sidebar.radio('Choose', ['About', 'Try the model', 'Try the visualizations', 'Explore the dataset'])

if option == 'Try the model':
    # st.title('This is a Work In Progress')
    # st.markdown('Check [this](https://github.com/Aditya-Ramachandran/cardiovascular-disease-detection) GitHub repository for updates')
    model_option = st.sidebar.selectbox('Choose Model', ['Ensemble Models', 'Neural Network'], key='model')
    # st.session_state
    # if st.session_state['model'] == 'Neural Network':
    #     NeuralNetwork()
    if st.session_state['model'] == 'Ensemble Models':
        ensemble_obj = EnsembleModels()    
        # ensemble_obj.display()
        ensemble_obj.predict()

if option == 'Try the visualizations':
        
        st.title('Visualizations')
        st.subheader('How to use')
        st.write('* Select the type of visualization from the sidebar')
        st.write('* Enter the values in the given input area (Press Enter to save)')
        
        st.markdown("<hr>", unsafe_allow_html=True)

        # https://youtu.be/92jUAXBmZyU -> Session State tut
        option_plot = st.sidebar.selectbox('Choose Visualization', ['Height vs Weight', 'Systolic BP vs Diastolic BP', 'Height vs systolic BP', 'Height vs diastolic BP', 'Weight vs systolic BP', 'Weight vs diastolic BP'],key="counter")
        # st.session_state
        if st.session_state['counter'] == 'Height vs Weight':
            st.subheader('Height vs Weight')
            height = st.number_input('Enter Height (in cm)')
            weight = st.number_input('Enter Weight (in kg)')
            st.pyplot(custom_scatter(df, height, weight, df['height'], df['weight'], 'height', 'weight', 'Height (cms)', 'Weight (kg)'))
            
        if st.session_state['counter'] == 'Systolic BP vs Diastolic BP':
            st.subheader('Systolic BP vs Diastolic BP')
            ap_hi = st.number_input('Enter Systolic BP')
            ap_lo = st.number_input('Enter Diastolic BP')
            st.pyplot(custom_scatter(ap_hi, ap_lo, df['ap_hi'], df['ap_lo'], 'ap_hi', 'ap_lo', 'Systolic blood pressure', 'Diastolic blood pressure'))
        
        # if st.session_state['counter'] == 'Age vs Height':
        #     st.subheader('Age vs Height')
        #     age = st.number_input('Enter your age')
        #     height = st.number_input('Enter your height')
        #     st.pyplot(custom_scatter(age, height, df['age'], df['height'], 'age', 'height', 'Age (years)', 'Height (cm)'))

        if st.session_state['counter'] == 'Height vs systolic BP':
            st.subheader('Height vs systolic BP')
            height = st.number_input('Enter height (in cms)')
            ap_hi = st.number_input('Enter Systolic BP')
            st.pyplot(custom_scatter(height, ap_hi, df['height'], df['ap_hi'], 'height', 'ap_hi', 'Height (cm)', 'Systolic Blood Pressure'))


        if st.session_state['counter'] == 'Height vs diastolic BP':
            st.subheader('Height vs diastolic BP')
            height = st.number_input('Enter height (in cms)')
            ap_lo = st.number_input('Enter Diastolic BP')
            st.pyplot(custom_scatter(height, ap_lo, df['height'], df['ap_lo'], 'height', 'ap_lo', 'Height (cm)', 'Diastolic Blood Pressure'))


        if st.session_state['counter'] == 'Weight vs systolic BP':
            st.subheader('Weight vs systolic BP')
            weight = st.number_input('Enter weight (in kg)')
            ap_hi = st.number_input('Enter Systolic BP')
            st.pyplot(custom_scatter(weight, ap_hi, df['weight'], df['ap_hi'], 'weight', 'ap_hi', 'Weight (kg)', 'Systolic Blood Pressure'))


        if st.session_state['counter'] == 'Weight vs diastolic BP':
            st.subheader('Weight vs diastolic BP')
            weight = st.number_input('Enter weight (in kg)')
            ap_lo = st.number_input('Enter Diastolic BP')
            st.pyplot(custom_scatter(weight, ap_lo, df['weight'], df['ap_lo'], 'weight', 'ap_lo', 'Weight (kg)', 'Diastolic Blood Pressure'))


if option == 'About':
    About.load_about()

explore_obj = Explore()

if option == 'Explore the dataset':
    explore_option = st.sidebar.selectbox('Choose', ['Filtering', 'Sorting'], key='explore')
    if st.session_state['explore'] == 'Filtering':
        st.header('Filtering')
        st.subheader('How to use')
        st.write('* Select the type of filtering')
        st.write('* For simple filter : select column name ->  the operator -> specify the value')
        st.write('* For multiple filters : Select the columns -> select operator for each column -> specfy the value for each column')
        st.write('* Press enter to get the result')
        st.write('* Filtered dataset can be downloaded from the link')
        st.markdown("<hr>", unsafe_allow_html=True)

        filter_option = st.radio('Select a filter type', ['Single Filter', 'Multiple Filters'], key='filter_op')
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if st.session_state['filter_op'] == 'Single Filter':
            st.subheader('Single Filter')
            explore_obj.filter()
        
        if st.session_state['filter_op'] == 'Multiple Filters':
            st.subheader('Multiple Filters')
            explore_obj.multiple_filters()


    if st.session_state['explore'] == 'Sorting':
        Explore.sorting()
