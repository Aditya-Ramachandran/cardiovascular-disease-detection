import streamlit as st

# st.title('Cardiovascular Disease Detection Using Python !')
# st.subheader('Major Project')

# functions for 'option'
def load_about():
    st.title('About the project')
    st.subheader('Introduction')
    st.write('Cardiovascular disease (CVD) refers to a group of disorders that affect the heart and blood vessels. It is a leading cause of death worldwide and can be caused by a variety of factors such as high blood pressure, high cholesterol, obesity, smoking, and a sedentary lifestyle. Some of the common forms of CVD include coronary artery disease, heart attacks, angina, stroke, and heart failure \n')

    st.subheader('Objective')
    st.write('The ultimate objective of CVD detection is to identify the disease as possible, prevent or manage its progression, and improve the overall health and well early as being of patients. By using various diagnostic tests and technologies, healthcare providers aim to provide personalized and effective treatment plans to patients, with the goal of reducing the incidence of CVD and its associated complications.')

    st.subheader('Methodology')
    st.markdown('#### Dataset')
    st.write('This project uses the Cardiovascular Disease dataset from Kaggle which contains 13 attributes and samples.')

    st.markdown('#### Classification Models')
    st.write('To improve the accuracy of cardiovascular disease diagnosis, we will propose an ensemble model based on machine learning algorithms, including XGBoost, K Nearest Neighbours, and Decision Trees, as well as a stacked model that combines an ML ensemble model with a Random Forest.We will utilize four machine learning techniques, namely Random Forest, KNN, Decision Trees, and XGBoost, in our project.')


st.sidebar.header('Cardiovascular Disease Detection Using Python')
option = st.sidebar.radio('Choose', ['About', 'Try the model'])

if option == 'Try the model':
    st.title('This is a WIP')

if option == 'About':
    load_about()
