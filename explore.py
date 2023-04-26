import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('Dataset/cardio_train_cleaned_1.csv')
df['age'] = round(df['age'])

# col1,col2,col3 = st.columns(3)

class Explore():

    def filter(self):
        explore_col =  st.selectbox('Select Column', [cols for cols in df.columns][1:])
        explore_operator = st.selectbox('Select Operator',['>', '>=','<','<=','=='])
        explore_num = st.number_input('Enter the value based on which you want to filter')
        self.__do_filter(df, explore_col, explore_operator, explore_num)
    
    def multiple_filters(self):
        selected_cols = st.multiselect("Select columns to filter on", [cols for cols in df.columns][1:])

        filters = []
        for col in selected_cols:
            op = st.selectbox(f"Select operator for {col}", options=["<", "<=", "==", ">=", ">"])
            val = st.number_input(f"Enter value for {col}", min_value=df[col].min(), max_value=df[col].max())
            filters.append((col, op, val))
        
        self.__do_multiple_filters(df, filters)
        

    def __do_multiple_filters(self, df, filters):
        def apply_filters(df, filters):
            for col, op, val in filters:
                df = df.query(f"{col} {op} {val}")
            return df

        if filters:
            filtered_df = apply_filters(df, filters)
            st.write(filtered_df)
        else:
            st.write(df)


    def __do_filter(self,dataframe_name, col_name, operator, number):
        if operator == '>':
            filtered_df = dataframe_name[dataframe_name[col_name] > number]
        
        if operator == '>=':
            filtered_df = dataframe_name[dataframe_name[col_name] >= number]

        if operator == '<':
            filtered_df = dataframe_name[dataframe_name[col_name] < number]

        if operator == '<=':
            filtered_df = dataframe_name[dataframe_name[col_name] <= number]

        if operator == '==':
            filtered_df = dataframe_name[dataframe_name[col_name] == number]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Number of rows :', filtered_df.shape[0])
        with col2:
            st.metric('Number of columns :', filtered_df.shape[1])
        with col3:
            st.metric('Total elements :', filtered_df.size)
        st.dataframe(filtered_df)
    

    def sorting():
        st.write('Sorting')
       


# explore_obj = Explore()

