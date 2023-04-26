import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# st.title('Cardiovascular Disease Detection Using Python !')
# st.subheader('Major Project')
st.set_page_config(page_title="Cardiovascular Disease Detection in Python")
st.set_option('deprecation.showPyplotGlobalUse', False)


df = pd.read_csv('Dataset/cardio_train_cleaned_1.csv')

########################################################################################################

# function for custom scatter plots
def custom_scatter(x,y, xx, yy, label1, label2, x_axis,y_axis):
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

#########################################################################################################


# functions for 'option'
def load_about():
    st.title('About the project')
    st.subheader('Introduction')
    st.write('* Cardiovascular disease (CVD) refers to a group of disorders that affect the heart and blood vessels. It is a leading cause of death worldwide and can be caused by a variety of factors such as high blood pressure, high cholesterol, obesity, smoking, and a sedentary lifestyle.') 
    st.write('*  Some of the common forms of CVD include coronary artery disease, heart attacks, angina, stroke, and heart failure \n')

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader('Objective')
    st.write('* The ultimate objective of CVD detection is to identify the disease as possible, prevent or manage its progression, and improve the overall health and well early as being of patients.')
    st.write('* By using various diagnostic tests and technologies, healthcare providers aim to provide personalized and effective treatment plans to patients, with the goal of reducing the incidence of CVD and its associated complications.')

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader('Approach')
    st.write('* The proposed method for predicting cardiovascular disease involves six phases, starting with the selection of a dataset, specifically the cardiovascular disease dataset for this study.') 
    st.write('* Pre-processing of the data involves several necessary steps before training the model.Feature extraction is then performed to assess the significance of the features.') 
    st.write('* To detect cardiovascular disease, the study employs four machine learning classifiers: Random Forest (RF), K-Nearest Neighbours (KNN), Decision Tree (DT), and Extreme Gradient Boosting (XGB).')

    st.markdown("<hr>", unsafe_allow_html=True)

    # html = """
    #     <div style="width: 500px;">
    #     <img src="images/height relation.png" style="float: left; padding-right: 10px;">
    #     <p>Here is some text that will wrap around the image.</p>
    #     </div>
    # """
    # st.markdown(html, unsafe_allow_html=True)

    st.subheader('Methodology')
    st.markdown('#### Dataset')
    st.write('* This project uses the Cardiovascular Disease dataset from Kaggle which contains 13 attributes and samples.')

    st.markdown('#### Data Pre-processing')
    st.write('* Before training a machine learning model, the data needs to be pre-processed to ensure its quality and that important information is available to the model.') 
    st.write('* This involves dealing with different characteristics of the dataset, such as missing values, outliers, and scaling. One way to pre-process the data is to eliminate outliers and normalize the data using a standard scaler, which adjusts the distribution of the data to have a mean of 0 and a standard deviation of 1.') 
    st.write('* The target labels are also transformed into numeric forms using a Label Encoder so that the machine can read them and make better predictions for disease diagnosis')

    st.markdown('#### Classification Models')
    st.write("* To improve the accuracy of cardiovascular disease diagnosis, we will propose an ensemble model based on machine learning algorithms, including XGBoost, K Nearest Neighbours, and Decision Trees, as well as a stacked model that combines an ML ensemble model with a Random Forest")
    st.write("* We will utilize four machine learning techniques, namely Random Forest, KNN, Decision Trees, and XGBoost, in our project.")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader('Flow Diagram')
    st.image('images/Untitled Diagram (1).jpg')


    # st.markdown("<hr>", unsafe_allow_html=True)

    # st.markdown('Check <a href="https://aditya-ramachandran.github.io/cardiovascular-disease-detection/">this</a> GitHub Repository for development updates.')

    st.markdown('Check [this](https://github.com/Aditya-Ramachandran/cardiovascular-disease-detection) GitHub repository for updates')


######################################################################################################### 


# function for model_option
def NeuralNetwork():
    st.write('NN')

def EnsembleModels():
    st.write('Ensemble Models')



st.sidebar.header('Cardiovascular Disease Detection Using Python')
option = st.sidebar.radio('Choose', ['About', 'Try the model', 'Try the visualizations'])

if option == 'Try the model':
    st.title('This is a Work In Progress')
    st.markdown('Check [this](https://github.com/Aditya-Ramachandran/cardiovascular-disease-detection) GitHub repository for updates')
    model_option = st.sidebar.selectbox('Choose Model', ['Ensemble Models', 'Neural Network'], key='model')
    # st.session_state
    if st.session_state['model'] == 'Neural Network':
        NeuralNetwork()
    if st.session_state['model'] == 'Ensemble Models':
        EnsembleModels()

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
            st.pyplot(custom_scatter(height, weight, df['height'], df['weight'], 'height', 'weight', 'Height (cms)', 'Weight (kg)'))
        
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



if option == 'About':
    load_about()
