#%%
from cProfile import label
from turtle import width
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import networkx as nx
import matplotlib.pylab as plt
from datetime import datetime

# https://plotly.com/python/network-graphs/
# https://pypi.org/project/d3graph/


dir=r'C:\Users\yosty\Desktop\Desktop_Folder\14 - git\timeSeriesDOTS\ploomber\dots'

st.title('Trade Dashboard')

@st.cache
def load_dots_data():
    data=pd.read_csv(os.path.join(dir, '00-data\dots\clean\dotsTimeSeries.csv'))
    data['period']=pd.to_datetime(data['period'])
    return(data)


@st.cache
def load_wto_data(country:str):
    data=pd.read(os.path.join(dir, f'00-data\wto\clean\hsm\{country}.csv'))
    return(data)

def topImporterChart():
    importerData=timeSeries.query('ReferenceArea==@exporterSelect').pivot_table(index='period', columns='CounterpartReferenceArea', values='value')
    topImporters=importerData.iloc[-nMonths:].melt().groupby('CounterpartReferenceArea')['value'].agg('sum').nlargest(nlarge)
    topImporterData=importerData[topImporters.index]
    topImportChart=px.line(topImporterData)
    topImportChart.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))
    topImportChart


timeSeries=load_dots_data()

exporter=timeSeries['ReferenceArea'].unique()
exporterSelect=st.sidebar.selectbox(label='Select Exporter', options=exporter)
importer=timeSeries[timeSeries['ReferenceArea']==exporterSelect]['CounterpartReferenceArea'].unique()
importerSelect=st.sidebar.selectbox(label='Select Importer', options=importer)
selectedTimeSeries=timeSeries.query('ReferenceArea==@exporterSelect and CounterpartReferenceArea==@importerSelect')

# start_time = st.sidebar.slider(
#      "Start Period Filter",
#      value=pd.to_datetime('2000-01-01'),
#      min_value=timeSeries['period'].min(),
#      max_value=timeSeries['period'].max())

# end_time = st.sidebar.slider(
#      "End Period Filter",
#      format="DD/MM/YY",
#      )

nMonths=st.sidebar.slider(label='Top n Importers for Trailing n Months', min_value=6, max_value=120, value=12)
nlarge=st.sidebar.slider(label='Top n Importers for Trailing n Months', min_value=1, max_value=10, value=5)


col1, col2= st.columns([50,50])

with col1:
    importerData=timeSeries.query('ReferenceArea==@exporterSelect').pivot_table(index='period', columns='CounterpartReferenceArea', values='value')
    topImporters=importerData.iloc[-nMonths:].melt().groupby('CounterpartReferenceArea')['value'].agg('sum').nlargest(nlarge)
    topImporterData=importerData[topImporters.index]
    topImportChart=px.line(topImporterData)
    topImportChart.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))
    st.header(body=f'{exporterSelect} Top Export Destinations')
    st.plotly_chart(topImportChart, use_container_width=True)


with col2:
    # st.dataframe(selectedTimeSeries)
    st.header(body=f'{exporterSelect} exports to {importerSelect}')
    linechart=px.line(selectedTimeSeries[['period', 'value']], x='period', y='value')
    st.plotly_chart(linechart, use_container_width=True)


    # G = nx.karate_club_graph()
    # fig, ax = plt.subplots()
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G,pos, with_labels=True)
    # st.pyplot(fig)




