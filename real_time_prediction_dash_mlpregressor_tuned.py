import wget
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash
from dash.dependencies import *
from dash import dcc
from dash import html
from sklearn.metrics import r2_score
from datetime import datetime
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
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

def array_to_df(array,colname):
	df = pd.DataFrame(array)
	df.columns =[colname]
	return df
def normalize(df, max_df, min_df,colname):
    normed = [(i - min_df)/(max_df - min_df) for i in df]
    normed_df = pd.DataFrame(normed)
    normed_df.columns = [colname]
    return normed_df

def denormalize(df, max_df, min_df,colname):
    denormed= [i*(max_df -min_df) + min_df for i in df]
    denormed_df = pd.DataFrame(denormed)
    denormed_df.columns = [colname]
    return denormed_df
def series_to_supervised(df,sequence_length):
	x=[]
	y=[]
	for i in range(sequence_length , len(df)-2*sequence_length):
		x.append(np.array(df[i-sequence_length:i]))
		y.append(np.array(df[i:i+sequence_length]))
	x= np.concatenate(x,axis = 0)
	y = np.concatenate(y,axis = 0)
	x = x.reshape(-1,1)
	y = y.reshape(-1,1)
	return x,y

def xgboost_forecast(train, X_test,sequence_length):
	params = {
        'hidden_layer_sizes': [(10,),(50,),(100,),(500,),(1000,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver' : ['sgd', 'adam'],
        'alpha': [0.001,0.0001,0.00001]


    }
	X_train, y_train = series_to_supervised(train,sequence_length)
	model = MLPRegressor(shuffle=False,early_stopping = True)
	reg = RandomizedSearchCV(estimator=model,
                             							param_distributions=params,
                             							scoring='neg_mean_absolute_error',
                             							n_iter=25,
                             							n_jobs=4,
                             							verbose=10)

	reg.fit(X_train, y_train)
	model.fit(X_train,y_train)
	bestmodel= reg.best_estimator_
	yhat= model.predict(X_test)
	bestyhat = bestmodel.predict(X_test)
	return yhat,bestyhat
def plot(yhat,bestyhat,y_test):
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=yhat, name='Predicted', mode='markers'))
	fig.add_trace(go.Scatter(y=y_test, name="True", mode='markers'))
	fig.add_trace(go.Scatter(y=bestyhat, name="Predicted after tuned", mode='markers'))
	fig.update_xaxes(title='Time')
	fig.update_yaxes(title='Price')
	fig.update_layout(title_text="Intraday price of EUR/USD")
	fig.show()
	return fig
@app.callback(Output('live-graph', 'figure'),[Input('graph-update', 'n_intervals')])
def getdata(*args,**kwargs):
	fromS = "EUR"
	toS = "USD"
	interval = "5min"
	apikey = "PU7EBSO3ACN7CDYI"
	url = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&outputsize=full&apikey={}&datatype=csv'.format(fromS,toS,interval,apikey)
	df = pd.read_csv(wget.download(url, out = "C:\\Temp\\"+str(fromS)+str(toS)+str(datetime.datetime.now().timestamp())+".csv"),
                                    header = 0,
                                    parse_dates= True,
                                    infer_datetime_format= True
                                    )
	dfd =df.iloc[:,4]
	max_df = dfd.max()
	min_df = dfd.min()
	dff= normalize(dfd, max_df,min_df,'close' )
	train,test = train_test_split(dff['close'], test_size=0.3,train_size=0.7,random_state=0,shuffle=False)
	sequence_length = 4
	X_test,y_test = series_to_supervised(test,sequence_length)
	yhat, bestyhat = xgboost_forecast(train,X_test,sequence_length)
	yhat = array_to_df(yhat,'close')
	yhat = denormalize(yhat['close'], max_df,min_df,'close')
	bestyhat = array_to_df(bestyhat, 'close')
	bestyhat = denormalize(bestyhat['close'],max_df,min_df,'close')
	y_test = y_test.reshape(-1,)
	y_test = array_to_df(y_test,'close')
	y_test = denormalize(y_test['close'], max_df,min_df,'close')
	print(f"r2 score before tuning {r2_score(yhat['close'],y_test['close'])}")
	print(f'r2 score after tuning{ r2_score(bestyhat["close"],y_test["close"])}')
	fig = plot(yhat['close'],bestyhat['close'],y_test['close'])
	return fig
if __name__ == '__main__':
	 app.run_server(debug=True, use_reloader=False)