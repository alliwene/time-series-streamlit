import streamlit as st

from datetime import timedelta
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR 

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf 
from tensorflow.keras.models import Model, Sequential, load_model

from use_functions import ( forecast_plot, rmse_results, fit_predict_lstm, 
                           fit_predict_var )

import warnings
warnings.simplefilter(action='ignore') 


def main():
    st.set_page_config(layout = "wide")
    page = st.sidebar.selectbox('Select Algorithm:', ['VAR','LSTM'])
    st.title('Predict Time Series Data')
    
    # load dataframe
    df = pd.read_csv('datasets/throughput_metrics.csv', parse_dates=['Time'], index_col='Time')
    df_print = pd.read_csv('datasets/throughput_metrics.csv', parse_dates=['Time'])
    
    st.markdown('#### Dataset:')
    st.write(df_print) 
    st.write(f'Dataset shape: {df.shape}')
    
    st.markdown('#### Dataset Statistics:')
    st.write(df.describe().T) 
    
    # missing value treatment
    for i in ['SiteB', 'SiteE']:
        df[i][df[i] == 0] = df[i].mean() 
    
    st.markdown("#### Test vs Prediction Plot:")
    
    
    if page == 'VAR':
        df_train, df_test, fc = fit_predict_var(df)
        fig = forecast_plot(df_train, df_test, fc, ['Train', 'Test', 'Predict'], 'VAR')
        st.plotly_chart(fig) 
        
    else:
        n_train_hours, inv_y, inv_yhat = fit_predict_lstm(df)
        df_train, df_test = df.iloc[:n_train_hours, :], df.iloc[n_train_hours:, :]
        fig = forecast_plot(df_train, df_test, inv_yhat, ['Train', 'Test', 'Predict'], 'LSTM') 
        st.plotly_chart(fig) 
    
if __name__ == "__main__":
    main() 