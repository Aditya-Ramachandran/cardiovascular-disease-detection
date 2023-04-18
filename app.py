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

    st.subheader('Approach')
    st.write('The proposed method for predicting cardiovascular disease involves six phases, starting with the selection of a dataset, specifically the cardiovascular disease dataset for this study. Pre-processing of the data involves several necessary steps before training the model. Feature extraction is then performed to assess the significance of the features. To detect cardiovascular disease, the study employs four machine learning classifiers: Random Forest (RF), K-Nearest Neighbours (KNN), Decision Tree (DT), and Extreme Gradient Boosting (XGB).')

    # html = """
    #     <div style="width: 500px;">
    #     <img src="images/height relation.png" style="float: left; padding-right: 10px;">
    #     <p>Here is some text that will wrap around the image.</p>
    #     </div>
    # """
    # st.markdown(html, unsafe_allow_html=True)

    st.subheader('Methodology')
    st.markdown('#### Dataset')
    st.write('This project uses the Cardiovascular Disease dataset from Kaggle which contains 13 attributes and samples.')

    st.markdown('#### Data Pre-processing')
    st.write('Before training a machine learning model, the data needs to be pre-processed to ensure its quality and that important information is available to the model. This involves dealing with different characteristics of the dataset, such as missing values, outliers, and scaling. One way to pre-process the data is to eliminate outliers and normalize the data using a standard scaler, which adjusts the distribution of the data to have a mean of 0 and a standard deviation of 1. The target labels are also transformed into numeric forms using a Label Encoder so that the machine can read them and make better predictions for disease diagnosis')

    st.markdown('#### Classification Models')
    st.write('To improve the accuracy of cardiovascular disease diagnosis, we will propose an ensemble model based on machine learning algorithms, including XGBoost, K Nearest Neighbours, and Decision Trees, as well as a stacked model that combines an ML ensemble model with a Random Forest.We will utilize four machine learning techniques, namely Random Forest, KNN, Decision Trees, and XGBoost, in our project.')

    


st.sidebar.header('Cardiovascular Disease Detection Using Python')
option = st.sidebar.radio('Choose', ['About', 'Try the model'])

if option == 'Try the model':
    st.title('This is a WIP')

if option == 'About':
    load_about()
