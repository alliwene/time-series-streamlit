import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib 
from statsmodels.tsa.vector_ar.var_model import VAR

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.simplefilter(action='ignore')


def main():
    st.set_page_config(layout = "wide")
    page = st.sidebar.selectbox('Select page', ['Forecast Data','Forecast Plot'])
    st.title('')
    
    df = pd.read_csv('datasets/throughput_metrics.csv', parse_dates=['Time'], index_col='Time')
    # missing value treatment
    df['SiteB'][df['SiteB'] == 0] = df['SiteB'].mean()
    
    # fit model and forecast
    model = VAR(df)  
    model_fit = model.fit() 
    yhat = model_fit.forecast(model_fit.y, steps=150)
    
    def forecast_to_df(data, forecast):
        ''' Convert forecast output array to dataframe using input data columns
        and input data last date '''
        date = data[-1:].index 
        future_date = []
        for i in range(len(forecast)): 
            date += timedelta(hours=1)
            future_date.append(date)
        yhat_df = pd.DataFrame(yhat, columns=data.columns, index=list(map(list, zip(*future_date))))
        # convert index values to datetime 
        idx = yhat_df.index.get_level_values(0).astype(str) 
        yhat_df.index = pd.to_datetime(idx)
        yhat_df.index.rename('Time', inplace=True)
        
        return yhat_df 
    
    yhat_df = forecast_to_df(df, yhat)
    
    # make forecast data similar to input data
    forecast_df = yhat_df.copy()
    forecast_df.reset_index(level=0, inplace=True)

    # st.markdown("## Forecast Data")
    # st.write(forecast_df) 
    
    # st.markdown("## Forecast Plot")
    fig = make_subplots(rows=6, cols=1, subplot_titles=df.columns,
                    vertical_spacing=0.05)

    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(name="actual", x=df.index, y=df[col],  
                                marker = dict(size = 10, 
                                            color = 'blue'),
                                textfont=dict(
                                    color='black',
                                    size=18,  
                                    family='Times New Roman')),
                    row=i+1, col=1)
        fig.add_trace(go.Scatter(name="forecast", x=yhat_df.index, y=yhat_df[col],  
                                marker = dict(size = 10, 
                                            color = 'red')),
                    row=i+1, col=1) 
        
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(autosize=False,
        width=1300,
        height=1500,
        )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    # remove duplicate legends in plot 
    # https://stackoverflow.com/questions/26939121/how-to-avoid-duplicate-legend-labels-in-plotly-or-pass-custom-legend-labels/62162555#62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    
    if page == 'Forecast Data':
        st.markdown("## Forecast Data")
        st.write(forecast_df) 
    else:
        st.markdown("## Forecast Plot")
        st.plotly_chart(fig) 
    
    
if __name__ == "__main__":
    main() 