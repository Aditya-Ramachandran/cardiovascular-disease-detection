import streamlit as st

# st.title('Cardiovascular Disease Detection Using Python !')
# st.subheader('Major Project')

# functions for 'option'
def load_about():
    st.title('About the project')

st.sidebar.header('Cardiovascular Disease Detection Using Python')
option = st.sidebar.radio('Choose', ['About', 'Try the model'])

if option == 'Try the model':
    st.title('This is a WIP')

if option == 'About':
    load_about()
