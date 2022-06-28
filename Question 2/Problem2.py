import pandas as pd
from math import sqrt 
import numpy as np 
import os 


class Strategy: 
    def __init__(self,data):
        
        self.data = data
        self.annual_days = 365

        '''
        Your strategy has to coded in this section 
        A sample strategy of moving average crossover has been provided to you. You can uncomment the same and run the code for checking data output 
        You strategy module should return a signal dataframe
        '''
        
    def strategy(self):
           
        # short_window = 15
        # long_window = 30
        # signal = self.data.rolling(short_window).mean() - self.data.rolling(long_window).mean()
        # signal.to_csv('signal.csv')
        # return signal
          
        signal = self.data
        for j in range(0, 10):
            stock_check = 0
            signal.iloc[0, j] = 0
            for i in range(1, self.data.shape[0] - 1):

                if stock_check == 0:
                    if((self.data.iloc[i,j]-self.data.iloc[i-1,j])/self.data.iloc[i-1,j] >= 0.02):
                        stock_check = 1
                        signal.iloc[i, j] = -self.data.iloc[i,j]
                    elif((self.data.iloc[i,j]-self.data.iloc[i-1,j])/self.data.iloc[i-1,j] <= -0.02):
                        stock_check = -1
                        signal.iloc[i, j] = self.data.iloc[i,j]
                    else:
                        signal.iloc[i, j] = 0
                elif stock_check == 1:
                    if((self.data.iloc[i,j]-self.data.iloc[i-1,j])/self.data.iloc[i-1,j] <= -0.01):
                        stock_check = 0
                        signal.iloc[i, j] = self.data.iloc[i,j]
                    else:
                        signal.iloc[i, j] = 0
                else:
                    if((self.data.iloc[i,j]-self.data.iloc[i-1,j])/self.data.iloc[i-1,j] >= 0.01):
                        stock_check = 0
                        signal.iloc[i, j] = -self.data.iloc[i,j]
                    else:
                        signal.iloc[i, j] = 0
            if stock_check == 1:
                signal.iloc[data.shape[0] - 1, j] = self.data.iloc[data.shape[0] - 1, j]
            else:
                signal.iloc[data.shape[0] - 1, j] = 0

        signal.to_csv('signal.csv')
        return signal
        '''
        This module computes the daily asset returns based on long/short position and stores them in a dataframe 
        '''
    def process(self):
        returns = self.data.pct_change()
        self.signal = self.strategy()
        self.position = self.signal.apply(np.sign)
        self.asset_returns = (self.position.shift(1)*returns)
        return self.asset_returns

        '''
        This module computes the overall portfolio returns, asset portfolio value and overall portfolio values 
        '''

    def portfolio(self):
        asset_returns = self.process()
        self.portfolio_return = asset_returns.sum(axis=1)
        self.portfolio = 100*(1+self.asset_returns.cumsum())
        self.portfolio['Portfolio'] = 100*(1+self.portfolio_return.cumsum())
        return self.portfolio

        '''
        This module computes the sharpe ratio for the strategy
        '''

    def stats(self):
        stats = pd.Series()
        self.index = self.portfolio()
        stats['Start'] = self.index.index[0]
        stats['End'] = self.index.index[-1]
        stats['Duration'] = pd.to_datetime(stats['End']) - pd.to_datetime(stats['Start'])
        annualized_return = self.portfolio_return.mean()*self.annual_days
        stats['Annualized Return'] = annualized_return
        stats['Annualized Volatility'] = self.portfolio_return.std()*sqrt(self.annual_days)
        stats['Sharpe Ratio'] = stats['Annualized Return'] / stats['Annualized Volatility']
        print(stats['Sharpe Ratio'])
        return stats
        
if __name__ == '__main__':

    """ 
    Function to read data from csv file 
    """
    data = pd.read_csv(os.path.join(os.getcwd(),'Data.csv'),index_col='Date')
    result = Strategy(data)
    res = result.stats()
    res.to_csv(os.path.join(os.getcwd(),'Result.csv'),header=False)





