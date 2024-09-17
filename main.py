import streamlit as st
import pandas as pd
import numpy as np 
from datetime import date 
import requests
from io import BytesIO 
import yfinance as yf 
import base64 
import plotly.express as px 
import plotly.graph_objects as go 
import cvxpy as cp 
class SessionState:
    def __init__(self,**kwargs):
        for key,val in kwargs.items():
            setattr(self,key,val) 
@st.cache_data(allow_output_mutation=True) 
def get_session():
    return SessionState(df=pd.DataFrame()) 
session_state = get_session() 
data = {
    "Name":['John',"Anna",'Peter','Linda'],
    'Age':[28,35,40,25]
}
df = pd.DataFrame(data) 
session_state.df = df 
def load_data_from_github(url):
    response = requests.get(url) 
    content = BytesIO(response.content) 
    data = pd.read_pickle(content) 
    return data 
#检索数据
def download_data(data,period='ly'):
    dfs = [] 
    if isinstance(data,dict):
        for name,ticker in data.items():
            ticker_obj = yf.Ticker(ticker) 
            hist = ticker_obj.history(period=period) 
            hist.columns = [f"{name}_{col}" for col in hist.columns] 
            hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
            dfs.append(hist) 
    elif isinstance(data, list):
        for ticker in data:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f'{ticker}_{col}' for col in hist.columns]  # Add prefix to the name
            hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
            dfs.append(hist)
# select box widget to choose timeframes
selected_timeframes = st.selectbox('Select Timeframe:', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=7)

from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models 
from pypfopt import expected_returns 
stock_data = {
    "AAPL":[0.1,0.05,0.08,0.12,0.07],
    "GOOG":[0.05,0.06,0.07,0.08,0.09],
    "MSFT":[0.06,0.04,0.08,0.07,0.05]
}
stocks_df = pd.DataFrame(stock_data) 

mu = expected_returns.mean_historical_return(stocks_df) 
S = risk_models.sample_cov(stocks_df) 
ef = EfficientFrontier(mu,S) 
weights = ef.max_sharpe() 
cleaned_weights = ef.clean_weights() 
print(cleaned_weights) 
ef.portfolio_performance(verbose=True) 
# 另一种方式
# 也可以使用求解器并实现 minimize 函数。这将为您节省大量时间，尽管在我的 16GB RAM 计算机上，10,000 次模拟只需大约一分钟即可完成。无论如何，应用 MVO 取决于：

# 估算资产的年化回报。
# 计算投资组合方差。
# 求投资组合中每对资产之间的协方差。
# 计算 Sharpe Ratio 并确定每个资产的最佳权重，使其最大化。
# 至少这些约束是必要的：权重之和必须等于 1 且大于 0（本教程不允许空头交易）。
# Risk-free rate
risk_free_rate = 0.05
# Number of assets
n_assets = len(expected_returns)
# Define variables
weights = cp.Variable(n_assets)
# Portfolio expected return
portfolio_return = expected_returns @ weights
# Portfolio volatility (standard deviation)
portfolio_volatility = cp.sqrt(cp.quad_form(weights, covariance_matrix))
# Portfolio Sharpe Ratio
portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
# Define objective function (maximize Sharpe Ratio)
objective = cp.Maximize(portfolio_sharpe_ratio)
# Define constraints
constraints = [
    cp.sum(weights) == 1,  # sum of weights equals 1 (fully invested)
    weights >= 0  # weights are non-negative
]
# You can add more constraints here, such as minimum and maximum weights,
# target return, etc.
# Solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

def logreturns(df):
    df.columns = df.columns.str.split('_').str[0] 
    log_returns = np.log(df) 
    log_returns = df.iloc[:, 0:].pct_change() 
    fig = px.line(log_returns,x=log_returns.index,y=log_returns.columns[0:].split('_')[0],labels={'value':'log'},title='Log Returns') 
    fig.update_layout(legend_title_text='Assets') 
    st.plotly_chart(fig) 
    return log_returns 
def return_over_time(df):
    return_df = pd.DataFrame()
    df.columns = df.columns.str.split('_').str[0]
    for col in df.columns:
        return_df[col] = df[col] / df[col].iloc[0] -1
        
    fig = px.line(return_df, x=return_df.index, y=return_df.columns[0:],
                  labels={'value': 'Returns to Date'},
                  title='returns')
    fig.update_layout(legend_title_text='Assets')
    st.plotly_chart(fig) # for streamlit plots
# 投资组合的回报

# 计算投资组合回报涉及将权重乘以平均对数回报。虽然每次模拟的权重都是随机生成的，但平均对数返回要求我们假设数据点的平均值。

# 在这种情况下，我们遵循的步骤是：

# 对所选时间帧的数据帧进行重采样
# 计算这些回报的平均值。
# 乘以交易日数（通常为 252 天）以获得年化回报
resampler = 'A' 
trading_days = 252 
annualized_returns = df.resample(resampler).last() 
annualized_returns = df.pct_change().apply(lambda x:np.log(1 + x)).mean() * trading_days 
simple_returns = np.exp(annualized_returns) - 1 
trading_days = 252
var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
sd = np.sqrt(var) # obtain the risk 
ann_sd = sd*np.sqrt(trading_days) # to scale the risk for any timeframe
def efficient_frontier(df, 
                       trading_days, 
                       risk_free_rate,
                       simulations= 1000, 
                       resampler='A'):
    
    # covariance matrix of the log returns
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    
    # lists to store the results
    portfolio_returns = [] 
    portfolio_variance = [] 
    portfolio_weights = [] 
    
    num_assets = len(df.columns) 
    for _ in range(simulations):
        # generating up to 1000 portfolio simulations
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights) # scaling weights to 1
        portfolio_weights.append(weights)
        # calculating the log returns
        returns = df.resample(resampler).last()
        returns = df.pct_change().apply(lambda x: np.log(1 + x)).mean() * trading_days
        annualized_returns = np.dot(weights, returns)
        portfolio_returns.append(annualized_returns)
        # portfolio variance
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(trading_days)
        portfolio_variance.append(ann_sd)
        # storing returns and volatitly in a dataframe
        data = {'Returns':portfolio_returns, 'Volatility':portfolio_variance}
    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol+' weight'] = [w[counter] for w in portfolio_weights]
    
    simulated_portfolios  = pd.DataFrame(data)
    simulated_portfolios['Sharpe_ratio'] = (simulated_portfolios['Returns'] - risk_free_rate) / simulated_portfolios['Volatility']
    
    return simulated_portfolios
def plot_efficient_frontier(simulated_portfolios, expected_sharpe, expected_return, risk_taken):
    simulated_portfolios = simulated_portfolios.sort_values(by='Volatility')
    
    # concatenating weights so we can hover on them as we select data points
    simulated_portfolios['Weights'] = simulated_portfolios.iloc[:, 2:-1].apply(
        lambda row: ', '.join([f'{asset}: {weight:.4f}' for asset, weight in zip(simulated_portfolios.columns[2:-1], row)]),
        axis=1
    )
    # creating the plot as a scatter graph
    frontier = px.scatter(simulated_portfolios, x='Volatility', y='Returns', width=800, height=600, 
                          title="Markowitz's Efficient Frontier", labels={'Volatility': 'Volatility', 'Returns': 'Return'},
                          hover_name='Weights')
    
    # getting the index of max Sharpe Ratio and painting in green
    max_sharpe_ratio_portfolio = simulated_portfolios.loc[simulated_portfolios['Sharpe_ratio'].idxmax()]
    frontier.add_trace(go.Scatter(x=[max_sharpe_ratio_portfolio['Volatility']], 
                                  y=[max_sharpe_ratio_portfolio['Returns']],
                                  mode='markers',
                                  marker=dict(color='green', size=10),
                                  name='Max Sharpe Ratio',
                                  text=max_sharpe_ratio_portfolio['Weights']))