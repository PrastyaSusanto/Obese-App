import pickle 
import streamlit as st
import pandas as pd

title = 'obese or not 💪'
subtitle = 'Predict obese or not with machine learning'


def main():
    st.set_page_config(layout="centered", page_icon='💪',
                       page_title='Obese or Not')
    st.title(title)
    st.write(subtitle)

    form = st.form("Data Input")
    gender = form.selectbox('Gender', ['Male','Female'])
    Age = form.number_input('How old are you?', min_value=1, max_value=100)
    family_history_with_overweight = form.selectbox('family history with overweight', ['Yes','No'])
    SMOKE = form.selectbox('are you smoking?',['Yes','No'])
    Height = form.number_input('Input your height (in meter)')
    Weight = form.number_input('Input your weight (in kilogram)')
    submit = form.form_submit_button('Predict your obesity level!')
    
    if submit:
        data = {
            'Gender':gender,
            'Age':Age,
            'Height':Height,
            'Weight':Weight,
            'family_history_with_overweight':family_history_with_overweight,
            'SMOKE':SMOKE,
            
            
        }
        data = pd.Series(data).to_frame(name=0).T
        data['Gender'] = data['Gender'].replace({'Male':0,'Female':1})
        data['family_history_with_overweight'] = data['family_history_with_overweight'].replace({'Yes':1,'No':0})
        data['SMOKE'] = data['SMOKE'].replace({'No':0,'Yes':1})
        with open('model.pkl','rb') as f:
            model = pickle.load(f)
        prediction = model.predict(data)[0]
        st.success('This is your state prediction : '+ prediction)
        
if __name__ == '__main__':
    main()
    
