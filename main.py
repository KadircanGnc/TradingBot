from data_loader import load_market_data
from strategy_factory import StrategyFactory
from config import Config
from strategies import MomentumStrategy, MovingAverageStrategy, RSIStrategy, BollingerBandsStrategy, MACDStrategy, AITradingStrategy
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
            self.strategy.execute({'price': prices})

if __name__ == "__main__":
    # Fetch data for a single company (e.g., "AAPL") for a longer period
    data = load_market_data(tickers=["AAPL"], period="2y")
    bot = TradingBot()

    # Set the desired strategy (AI Strategy)
    ai_strategy = AITradingStrategy()
    bot.set_strategy(ai_strategy)

    # Run the backtest with the selected strategy
    backtest = Backtest(ai_strategy, data)
    backtest.run()