import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import datetime

# Portfolio Optimization
def alpha_beta(x: pd.Series, y: pd.Series, verbose=False) -> list:
    """
    Get alpha and beta indicators
    :param x: Time series data to compare against, typically the market e.g. 'SPY'
    :param y: Time series data of the stock, e.g. 'TSLA'
    :return: list with [alpha, beta] 
    https://www.investopedia.com/articles/investing/092115/alpha-and-beta-beginners.asp
    """
    beta = (((x.mean()*y.mean()) - (x*y).mean()) / ((x.mean()**2) - (x**2).mean()))
    alpha = y.mean() - beta*x.mean()

    if verbose:
        print(f'offset:alpha = {alpha}, slope:beta = {beta}')

    return [alpha, beta]


def plot_ab(df: pd.DataFrame, market: str, ticker: str, size:tuple = (7, 7)) -> None:
    """
    Plot alpha vs beta indicators
    :param df: dataframe with stocks data
    :param market: string of ticker to be used for comparison
    :param ticker: string of ticker to analyze
    :param size: tuple for the size of the plot
    """
    x = df[market]
    y = df[ticker]
    alpha, beta = np.round(alpha_beta(x, y), 3)
    
    plt.figure(figsize=size)
    plt.title(f'{ticker} vs. {market} - alpha: {alpha}, beta: {beta}')
    plt.scatter(x, y)
    plt.plot(x, x*beta, color='r')
    plt.xlabel(f'{market}')
    plt.ylabel(f'{ticker}')
    plt.show()   

# Market goal optimization functions
def sortino(allocation:list, portfolio:pd.DataFrame, drf:float = 0.0002, samples:int = 252) -> float:
    """
    TODO: review implementation
    Get sortino ratio
    :param allocation: weights to be allocated to the stock
    :param portfolio: dataframe with timeseries data of multiple stocks
    :param drf: gains from a risk free investment
    :param samples: time to consider, 252 days / 1 year of trading
    :return: negative of sortino ratio (optimizer minimizes the function)
    """
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*np.sqrt(samples)*(portfolio.mean()-drf)/portfolio[portfolio<0].std()


def sharpe(allocation:list, portfolio:pd.DataFrame, drf:float = 0.0002, samples:int = 252) -> float:
    """
    Get sharpe ratio
    :param allocation: weights to be allocated to the stock
    :param portfolio: dataframe with timeseries data of multiple stocks
    :param drf: gains from a risk free investment
    :param samples: time to consider, 252 days / 1 year of trading
    :return: negative of sharpe ratio (optimizer minimizes the function)
    """
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*np.sqrt(samples)*(portfolio.mean()-drf)/portfolio.std()


def calmar(allocation:list, portfolio:pd.DataFrame, drf:float = 0.0002, samples:int = 252) -> float:
    """
    Get sharpe ratio
    :param allocation: weights to be allocated to the stock
    :param portfolio: dataframe with timeseries data of multiple stocks
    :param drf: gains from a risk free investment
    :param samples: time to consider, 252 days / 1 year of trading
    :return: negative of calmar ratio (optimizer minimizes the function)
    """
    portfolio = (portfolio*allocation).sum(axis=1)
    return (-1*np.sqrt(samples)*(portfolio.mean()-drf) / abs(portfolio.min()))


def profit(allocation:list, portfolio:pd.DataFrame) -> float:
    """
    TODO: review implementation
    Calculates the profit (or loss) from a portfolio
    :param allocation: weights to be allocated to the stock
    :param portfolio: dataframe with timeseries data of multiple stocks
    :return: negative profit (optimizer minimizes the function)
    """
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*portfolio.sum() 


def alpha_f(allocation:list, portfolio:pd.DataFrame, market:pd.Series) -> float:
    """
    Calculates the alpha for a portfolio
    :param allocation: weights to be allocated to the stock
    :param portfolio: dataframe with timeseries data of multiple stocks
    :return: negative alpha (optimizer minimizes the function)
    """
    portfolio = (portfolio*allocation).sum(axis=1)
    alpha, _ = alpha_beta(market, portfolio)
    return -alpha


def beta_f(allocation:list, portfolio:pd.DataFrame, market:pd.Series) -> float:
    """
    Calculates the beta for a portfolio
    :param allocation: weights to be allocated to the stock
    :param portfolio: dataframe with timeseries data of multiple stocks
    :return: negative beta (optimizer minimizes the function)
    """
    portfolio = (portfolio*allocation).sum(axis=1)
    _, beta = alpha_beta(market, portfolio)
    return -beta


def portfolio_optimization(alg, portfolio: pd.DataFrame, method:str = 'SLSQP', bench:str = 'SPY') -> list:
    """
    Optimizes the portfolio for a given market indicator function finding 
    the optimal allocation of funds for the available stocks 
    :param alg: a function to optimize (minimize)
    :param portfolio: dataframe with timeseries data of multiple stock
    :param method: optimization method to chosse from 
                    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    :param bench: benchmark to compare the stock against, used with alpha and beta optimization.
    :return: list with weights or allocation proportion to each stock
    """
    cons = ({'type':'eq', 'fun':lambda x: 1-sum(x)})

    market_av = bench in portfolio.columns

    if market_av:
        market = portfolio[bench]
        portfolio.drop(bench, axis=1, inplace=True)

    n = portfolio.shape[1]
    
    allocation = [1/n]*n
    bounds = [(0, 1) for _ in range(n)]
    
    if alg.__name__ not in ('alpha_f', 'beta_f'):
        return spo.minimize(alg, allocation, args=(portfolio,), method=method, options={'disp': True}, bounds=bounds, constraints=cons)
    else:
        return spo.minimize(alg, allocation, args=(portfolio, market,), method=method, options={'disp': True}, bounds=bounds, constraints=cons)


def optimize(alg, stock:pd.DataFrame, dates:tuple, bench:str = 'SPY', eps:float=0.001) -> list:
    """
    :param alg: a function to optimize (minimize)
    :param stock: dataframe with historic stock data
    :param dates: pair of (start, end) strings with the date to start / end optimization
    :param bench: benchmark to compare the stock against, used with alpha and beta optimization
    :param eps: minimum proportion to assign to a stock
    :return: dataframe of proportions to be assigned to stocks and the ratio for them     
    """
    index = ((stock.index >= dates[0]) &  (stock.index <= dates[1]))

    portfolio = stock[index].pct_change().fillna(0)
    market = portfolio[bench]
    result = portfolio_optimization(alg, portfolio)
    
    assigned = []

    for i, j in zip(portfolio.columns.values, result.x):
        assigned.append([i,j])
    
    assigned = pd.DataFrame(assigned, columns=['ticker', 'assigned'])
    assigned.sort_values(by='assigned', ascending=False, inplace=True)
    assigned = assigned[assigned.assigned > 0]
    
    if alg.__name__ in ('alpha_f', 'beta_f'):
        ratio = np.round(-alg(result.x, portfolio, market), 3)
    else:
        ratio = np.round(-alg(result.x, portfolio), 5)
    
    assigned = assigned[assigned.assigned >= eps].sort_values(by='assigned', ascending=False)
    print(f'\n{alg.__name__}: {ratio}')
    return [assigned, ratio]

# Short Selling
def moving_average(stock: pd.DataFrame, ticker:str, periods:list) -> pd.DataFrame:
    """
    Create moving averages for a particular stock
    :param stock: dataframe with stocks
    :param ticker: stock to create ma for
    :param periods: list of number of time periods to calculate the ma for
    :return: dataframe with stock and its moving averages
    """
    temp = stock[[ticker]].copy()
    
    for n in periods:
        temp[f'ma_{n}'] = temp[ticker].rolling(n).mean()
    
    return temp 


def crosses(stock: pd.DataFrame, ticker:str, periods:list) -> pd.DataFrame:
    """
    Checks when a faster ma crosses upward a slower ma
    :param stock: dataframe with stock's moving averages
    :param ticker: stock to create ma for
    :param periods: list of number of time periods to calculate the ma for
    :return: dataframe with pair comparisons between moving averages
    """
    
    for i in range(len(periods)-1):
        for j in range(i+1, len(periods)):
            stock[f'cross_{periods[i]}_{periods[j]}'] = stock[f'ma_{periods[i]}'] > stock[f'ma_{periods[j]}']
        
    return stock


def bollinger(df: pd.DataFrame, ticker:str, n:int, stds:float) -> pd.DataFrame:
    """
    """
    temp = df[[ticker]].copy()
    temp[f'ma_{n}'] = temp[ticker].rolling(n).mean()
    temp[f'std_{n}'] = temp[ticker].rolling(n).std()
    temp[f'bb_lo_{n}'] = temp[f'ma_{n}'] - stds * temp[f'std_{n}']
    temp[f'bb_hi_{n}'] = temp[f'ma_{n}'] + stds * temp[f'std_{n}']
    return temp


class buySell(object):
    """
    Class to test a buy/sell strategy based on a time series binary signal
    """
    def __init__(self):
        """
        Initialize the function and variables, balance starts at 0
        flag is False (flag states if we have bought or not) and trans
        is the list of all transactions made, negative are buys and
        positive are sells
        """
        self.balance = 0
        self.flag = True
        self.trans = []
    
    def buy(self, amnt):
        self.balance -= amnt
        self.flag = True
        self.trans.append(-amnt)
        
    def sell(self, amnt):
        self.balance += amnt
        self.flag = False
        self.trans.append(amnt)
        
    def transactions(self, signal, amnt):
        if not self.flag:
            if signal:
                self.buy(amnt)
        else:
            if not signal:
                self.sell(amnt)