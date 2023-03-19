#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Path assuming "Asharq.csv" is in the same directory as this python file
tv_df = pd.read_csv("Asharq.csv", skiprows=10)

# Data Manipulation
tv_df.rename(columns={'As Selected':'Channel','As Selected.1':'Date'}, inplace=True)
tv_df.drop(['Heading'], axis=1, inplace=True)

# Below line to exclude blank lines at the end of the CSV file
tv_df = tv_df[tv_df['19:30:00'].notna()]

# Unpivotting the data
df_unpivoted = tv_df.melt(id_vars=['Channel','Date'], var_name='Time', value_name='Reach%')

# Adding date time column
df_unpivoted['Date Time'] = df_unpivoted['Date'] + ' ' + df_unpivoted['Time']
dates_list = list(df_unpivoted['Date'].unique())
# df_unpivoted.drop(['Date'], axis=1, inplace=True)
df_unpivoted.drop(['Time'], axis=1, inplace=True)

# Changing data types
df_unpivoted = df_unpivoted.astype({"Channel": str, "Reach%": float})
df_unpivoted['Date Time']= pd.to_datetime(df_unpivoted['Date Time'])

# Generating the graphs per day of the week
for date in dates_list:
    df_pivoted = df_unpivoted[df_unpivoted['Date'] == date].set_index(['Date Time','Channel']).unstack()['Reach%']
    
    df_pivoted.plot(title = '% Reach by Channel for '+date, figsize=(20,10))


# In[2]:


sum_reach = df_unpivoted.groupby('Channel')['Reach%'].sum().sort_values(ascending=False).reset_index()

# Generating the graph for sum of viewership reach over the 9 days
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(sum_reach['Channel'], sum_reach['Reach%'])
plt.title('Sum of % Reach by Channel for 9 Days')
plt.xlabel('Channels')
plt.ylabel('% Reach')
plt.show()


# In[3]:


daily_reach = df_unpivoted.groupby(['Date','Channel'])['Reach%'].mean().reset_index()

# Generating the graph for daily reach percentage
daily_reach.pivot(index='Date', columns='Channel', values='Reach%').plot(title = 'Daily % Reach by Channel', figsize=(20,10))


# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Get the data
tv_df = pd.read_csv("Asharq.csv", skiprows=10)

# Data Manipulation
tv_df.rename(columns={'As Selected':'Channel','As Selected.1':'Date'}, inplace=True)
tv_df.drop(['Heading'], axis=1, inplace=True)

# Below line to exclude blank lines at the end of the CSV file
tv_df = tv_df[tv_df['19:30:00'].notna()]

# Unpivotting the data
df_unpivoted = tv_df.melt(id_vars=['Channel','Date'], var_name='Time', value_name='Reach%')

# Adding date time column
df_unpivoted['Date Time'] = df_unpivoted['Date'] + ' ' + df_unpivoted['Time']
dates_list = list(df_unpivoted['Date'].unique())
# df_unpivoted.drop(['Date'], axis=1, inplace=True)
df_unpivoted.drop(['Time'], axis=1, inplace=True)

# Changing data types
df_unpivoted = df_unpivoted.astype({"Channel": str, "Reach%": float})
df_unpivoted['Date Time']= pd.to_datetime(df_unpivoted['Date Time'])


# Set up the Dash App
app = dash.Dash()

# Create a Dropdown to select the Date
dates_dropdown = dcc.Dropdown(
    id='dates_dropdown',
    options=[{'label': date, 'value': date} for date in dates_list],
    value=dates_list[0]
)

# Create a Graph component
graph = dcc.Graph(id='tv_reach_graph')

# Define the layout of the App
app.layout = html.Div(children=[
    html.H1('Asharq TV Reach'),
    dates_dropdown,
    graph
])

# Define the callback to update the Graph
@app.callback(
    dash.dependencies.Output('tv_reach_graph', 'figure'),
    [dash.dependencies.Input('dates_dropdown', 'value')])
def update_graph(selected_date):
    
    # Generate the graph data
    df_pivoted = df_unpivoted[df_unpivoted['Date'] == selected_date].set_index(['Date Time','Channel']).unstack()['Reach%']
    
    # Create the graph
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for channel in df_pivoted.columns:
        fig.add_trace(go.Scatter(x=df_pivoted.index, y=df_pivoted[channel], name=channel),secondary_y=False)
    fig.update_layout(title="% Reach by Channel for "+selected_date)

    return fig


# Run the Dash App
if __name__ == '__main__':
    app.run_server()
server = app.server

# In[ ]:




