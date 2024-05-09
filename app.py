import pandas as pd
import numpy as np
import joblib
import streamlit as st
import DateTime as dt


def main():
    st.title("Gail Share Price Prediction")  # Corrected title for clarity



    columns = {
        'date': st.text_input('Date', value='06-05-24'),
        'open': st.number_input('Open', value=0.00),
        'high': st.number_input('High', value=0.00),
        'low': st.number_input('Low', value=0.00),
        'volume': st.number_input('Volume in million', value=0.00)
    }







    if st.button('Predict'):
        st.write('Model is Predicting...')

        # Create a DataFrame from user input
        Data = pd.DataFrame(columns, index=[0])

        Data['date'] = pd.to_datetime(Data['date'])
        Data['year'] = Data['date'].dt.year
        Data['month'] = Data['date'].dt.month
        Data['day'] = Data['date'].dt.day
        Data['day_of_week'] = Data['date'].dt.dayofweek
        Data['day_of_year'] = Data['date'].dt.dayofyear
        Data['quarter'] = Data['date'].dt.quarter
        Data['day_name'] = Data['date'].dt.day_name()

        from sklearn.preprocessing import LabelEncoder
        Data = Data.drop(['date'], axis=1)
        Data['day_name'] = LabelEncoder().fit_transform(Data['day_name'])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # Data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(Data[['open', 'high', 'low', 'close', 'volume']])
        Data['volume'] = Data['volume'].values * 1000000

        Data['volume'] = scaler.fit_transform(Data['volume'].values.reshape(-1, 1))




        # Load the model
        try:
            model = joblib.load('gail_model.pkl')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return  # Exit function gracefully

        # Predict using the model
        st.subheader('Predicting...')
        prediction = model.predict(Data)
        try:
            prediction = prediction
            st.write(f"Close price : {prediction} INR")
        except Exception as e:
            st.error(f"Error predicting: {e}")
            return
        pass


if __name__ == '__main__':
    main()
