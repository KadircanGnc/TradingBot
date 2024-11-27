from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from data_loader import create_features

class TradingStrategy(ABC):
    @abstractmethod
    def execute(self, data):
        # Template method
        pass

class MomentumStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        signals = []
        for i in range(1, len(prices)):
            momentum = prices[i] - prices[i - 1]
            if momentum > 0:
                signals.append(1)  # Buy signal
            else:
                signals.append(-1)  # Sell signal
        return signals

class MovingAverageStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        if prices.ndim != 1:
            prices = prices.flatten()
        
        short_window = 5
        long_window = 10

        if len(prices) < long_window:
            return []

        short_mavg = np.convolve(prices, np.ones(short_window)/short_window, mode='valid')
        long_mavg = np.convolve(prices, np.ones(long_window)/long_window, mode='valid')

        min_length = min(len(short_mavg), len(long_mavg))
        short_mavg = short_mavg[-min_length:]
        long_mavg = long_mavg[-min_length:]

        signals = []
        for i in range(min_length):
            if short_mavg[i] > long_mavg[i]:
                signals.append(1)  # Buy signal
            else:
                signals.append(-1)  # Sell signal
        return signals

class RSIStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        window = 14
        if len(prices) < window:
            return []

        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100./(1. + rs)

        signals = []
        for i in range(window, len(prices)):
            delta = deltas[i - 1]  # The diff is 1 shorter

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(window - 1) + upval)/window
            down = (down*(window - 1) + downval)/window

            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

            if rsi[i] < 30:
                signals.append(1)  # Buy signal
            elif rsi[i] > 70:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # No signal
        return signals

class BollingerBandsStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        window = 20
        if len(prices) < window:
            return []

        rolling_mean = np.convolve(prices, np.ones(window)/window, mode='valid')
        rolling_std = np.std(prices[:window])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        signals = []
        for i in range(len(rolling_mean)):
            if prices[i + window - 1] < lower_band[i]:
                signals.append(1)  # Buy signal
            elif prices[i + window - 1] > upper_band[i]:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # No signal
        return signals

class MACDStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        short_window = 12
        long_window = 26
        signal_window = 9

        if len(prices) < long_window + signal_window - 1:
            return []

        short_ema = np.convolve(prices, np.ones(short_window)/short_window, mode='valid')
        long_ema = np.convolve(prices, np.ones(long_window)/long_window, mode='valid')
        macd = short_ema[-len(long_ema):] - long_ema
        signal = np.convolve(macd, np.ones(signal_window)/signal_window, mode='valid')

        if len(macd) < signal_window or len(signal) < signal_window:
            return []

        signals = []
        for i in range(len(signal)):
            if macd[i + signal_window - 1] > signal[i]:
                signals.append(1)  # Buy signal
            elif macd[i + signal_window - 1] < signal[i]:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # No signal
        return signals

class EMACrossoverStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        short_window = 12
        long_window = 26

        if len(prices) < long_window:
            return []

        short_ema = self.ema(prices, short_window)
        long_ema = self.ema(prices, long_window)

        signals = []
        for i in range(len(long_ema)):
            if short_ema[i + long_window - short_window] > long_ema[i]:
                signals.append(1)  # Buy signal
            elif short_ema[i + long_window - short_window] < long_ema[i]:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # No signal
        return signals

    def ema(self, prices, window):
        ema = [sum(prices[:window]) / window]
        multiplier = 2 / (window + 1)
        for price in prices[window:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        return ema

class AITradingStrategy(TradingStrategy):
    def __init__(self):
        self.model = LinearRegression()
        self.momentum_strategy = MomentumStrategy()
        self.moving_average_strategy = MovingAverageStrategy()
        self.rsi_strategy = RSIStrategy()
        self.bollinger_bands_strategy = BollingerBandsStrategy()
        self.ema_crossover_strategy = EMACrossoverStrategy()

    def train(self, prices):
        X, y = create_features(prices)
        momentum_signals = self.momentum_strategy.execute({'price': prices})
        moving_average_signals = self.moving_average_strategy.execute({'price': prices})
        rsi_signals = self.rsi_strategy.execute({'price': prices})
        bollinger_bands_signals = self.bollinger_bands_strategy.execute({'price': prices})
        ema_crossover_signals = self.ema_crossover_strategy.execute({'price': prices})

        # Ensure the signals are the same length as the features
        min_length = min(len(X), len(momentum_signals), len(moving_average_signals), len(rsi_signals), len(bollinger_bands_signals), len(ema_crossover_signals))
        X = X[:min_length]
        y = y[:min_length]
        momentum_signals = momentum_signals[:min_length]
        moving_average_signals = moving_average_signals[:min_length]
        rsi_signals = rsi_signals[:min_length]
        bollinger_bands_signals = bollinger_bands_signals[:min_length]
        ema_crossover_signals = ema_crossover_signals[:min_length]

        # Add the signals as additional features
        X = np.hstack((X, np.array(momentum_signals).reshape(-1, 1), np.array(moving_average_signals).reshape(-1, 1), np.array(rsi_signals).reshape(-1, 1), np.array(bollinger_bands_signals).reshape(-1, 1), np.array(ema_crossover_signals).reshape(-1, 1)))
        self.model.fit(X, y)

    def execute(self, data):
        prices = data['price']
        if len(prices) < 26:  # Ensure enough data for all strategies
            print("Not enough data to make predictions")
            return

        self.train(prices[:-1])
        momentum_signals = self.momentum_strategy.execute({'price': prices[-26:]})
        moving_average_signals = self.moving_average_strategy.execute({'price': prices[-26:]})
        rsi_signals = self.rsi_strategy.execute({'price': prices[-26:]})
        bollinger_bands_signals = self.bollinger_bands_strategy.execute({'price': prices[-26:]})
        ema_crossover_signals = self.ema_crossover_strategy.execute({'price': prices[-26:]})

        # Print the lengths of the signal lists for debugging
        print(f"Length of momentum signals: {len(momentum_signals)}")
        print(f"Length of moving average signals: {len(moving_average_signals)}")
        print(f"Length of RSI signals: {len(rsi_signals)}")
        print(f"Length of Bollinger Bands signals: {len(bollinger_bands_signals)}")
        print(f"Length of EMA Crossover signals: {len(ema_crossover_signals)}")

        # Ensure each signal list has enough elements
        if not (momentum_signals and moving_average_signals and rsi_signals and bollinger_bands_signals and ema_crossover_signals):
            print("Not enough signals to make predictions")
            return

        # Create the feature vector for prediction
        X_pred = np.hstack((prices[-5:], momentum_signals[-1], moving_average_signals[-1], rsi_signals[-1], bollinger_bands_signals[-1], ema_crossover_signals[-1])).reshape(1, -1)
        prediction = self.model.predict(X_pred)[0]
        print(f"Predicted next price: {prediction}")

        if prediction > prices[-1]:
            print("Buy signal based on AI strategy")
        else:
            print("Sell signal based on AI strategy")