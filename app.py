from flask import Flask, render_template, request
from data_loader import load_market_data
from strategy_factory import StrategyFactory
from strategies import MomentumStrategy, MovingAverageStrategy, RSIStrategy, BollingerBandsStrategy, EMACrossoverStrategy
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    ticker = request.form['ticker']
    strategy_type = request.form['strategy']
    
    # Load market data
    data = load_market_data(tickers=[ticker], period="3mo")
    
    # Get the selected strategy
    strategy = StrategyFactory.get_strategy(strategy_type)
    
    # Run the backtest
    bot = TradingBot()
    bot.set_strategy(strategy)
    backtest = Backtest(strategy, data)
    signals, prices = backtest.run_single(ticker)
    
    # Plot the signals
    img = io.BytesIO()
    plot_signals(ticker, prices, signals, img)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('result.html', plot_url=plot_url)

def plot_signals(ticker, prices, signals, img):
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
    plt.savefig(img, format='png')
    plt.close()

class TradingBot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingBot, cls).__new__(cls)
            cls._instance.strategy = None
        return cls._instance

    def set_strategy(self, strategy):
        self.strategy = strategy

class Backtest:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run_single(self, ticker):
        prices = self.data[ticker]
        signals = self.strategy.execute({'price': prices})
        return signals, prices

if __name__ == '__main__':
    app.run(debug=True)