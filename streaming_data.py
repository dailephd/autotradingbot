'''API Parameters
❚ Required: function
The time series of your choice. In this case, function=TIME_SERIES_INTRADAY_EXTENDED
❚ Required: symbol
The name of the equity of your choice. For example: symbol=IBM
❚ Required: interval
Time interval between two consecutive data points in the time series.
The following values are supported: 1min, 5min, 15min, 30min, 60min
❚ Required: slice
Two years of minute-level intraday data contains over 2 million data points, which can take up to Gigabytes of memory.
To ensure optimal API response speed, the trailing 2 years of intraday data is evenly divided into 24 "slices" - year1month1, year1month2, year1month3, ..., year1month11, year1month12, year2month1, year2month2, year2month3, ..., year2month11, year2month12. Each slice is a 30-day window, with year1month1 being the most recent and year2month12 being the farthest from today. By default, slice=year1month1.
❚ Optional: adjusted
By default, adjusted=true and the output time series is adjusted by historical split and dividend events. Set adjusted=false to query raw (as-traded) intraday values.
❚ Required: apikey
Your API key. Claim your free API key here.
Examples
To ensure optimal API response time, this endpoint uses the CSV format which is more memory-efficient than JSON.

Split/dividend-adjusted 15min intraday data for IBM covering the most recent 30 days (slice=year1month1):
https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month1&apikey=demo

Split/dividend-adjusted 15min intraday data for IBM covering the most recent day 31 through day 60 (slice=year1month2):
https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=15min&slice=year1month2&apikey=demo

Raw (as-traded) 60min intraday data for IBM covering the most recent day 61 through day 90 (slice=year1month3):
https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=60min&slice=year1month3&adjusted=false&apikey=demo '''
import wget
import pandas as pd
import plotly.express as px
import dash
from dash.dependencies import *
from dash import dcc
from dash import html
import datetime
app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div([
        html.H4('EUR/USD'),
        dcc.Graph(id='live-graph',animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000*300, # in milliseconds
            n_intervals=0
            )
        ])
    )

@app.callback(Output('live-graph', 'figure'),[Input('graph-update', 'n_intervals')])
def getdata(n):
    fromS = "EUR"
    toS = "USD"
    interval = "5min"
    apikey = "PU7EBSO3ACN7CDYI"
    url = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&apikey={}&datatype=csv'.format(fromS,toS,interval,apikey)
    df = pd.read_csv(wget.download(url, out = "C:\\Temp\\"+str(fromS)+str(toS)+str(datetime.datetime.now().timestamp())+".csv"))
    fig = px.scatter(df,
                            x = 'timestamp',
                            y = 'open',
                            width = 600,
                            height = 400,
                            template = "plotly_dark" )
    fig.update_xaxes(title = 'Time')
    fig.update_yaxes(title = 'Open')
    fig.update_layout(title_text = "Open Price of " + str(fromS)+"/" +str(toS))
    return fig
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
