from data_loader import load_market_data
from strategy_factory import StrategyFactory
from config import Config
from strategies import MomentumStrategy, MovingAverageStrategy, RSIStrategy, BollingerBandsStrategy, EMACrossoverStrategy
import numpy as np
import matplotlib.pyplot as plt

class TradingBot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingBot, cls).__new__(cls)
            cls._instance.strategy = None
        return cls._instance

    def set_strategy(self, strategy):
        self.strategy = strategy

    def trade(self, data):
        if self.strategy is not None:
            self.strategy.execute(data)
        else:
            print("No strategy set")

class Backtest:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run(self):
        print("Running backtest...")
        for ticker, prices in self.data.items():
            print(f"Backtesting for {ticker}")
            signals = self.strategy.execute({'price': prices})
            self.plot_signals(ticker, prices, signals)

    def plot_signals(self, ticker, prices, signals):
        buy_signals = [i for i, signal in enumerate(signals) if signal == 1]
        sell_signals = [i for i, signal in enumerate(signals) if signal == -1]

        plt.figure(figsize=(14, 7))
        plt.plot(prices, label='Price')
        plt.scatter(buy_signals, prices[buy_signals], marker='^', color='g', label='Buy Signal', alpha=1)
        plt.scatter(sell_signals, prices[sell_signals], marker='v', color='r', label='Sell Signal', alpha=1)
        plt.title(f'Backtesting {ticker}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Fetch data for multiple companies (e.g., "AAPL", "MSFT", "GOOGL", "AMZN", "NFLX") for a longer period
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NFLX"]
    data = load_market_data(tickers=tickers, period="3mo")
    bot = TradingBot()

    # Set the desired strategy (e.g., Momentum Strategy)
    momentum_strategy = MomentumStrategy()
    bot.set_strategy(momentum_strategy)

    # Run the backtest with the selected strategy
    backtest = Backtest(momentum_strategy, data)
    backtest.run()