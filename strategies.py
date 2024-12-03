from abc import ABC, abstractmethod
import numpy as np

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

class EMACrossoverStrategy(TradingStrategy):
    def execute(self, data):
        prices = data['price']
        short_window = 12
        long_window = 26

        if len(prices) < long_window:
            return []

        short_ema = self.ema(prices, short_window)
        long_ema = self.ema(prices, long_window)

        # Ensure the short_ema and long_ema arrays are aligned correctly
        short_ema = short_ema[-len(long_ema):]

        signals = []
        for i in range(len(long_ema)):
            if short_ema[i] > long_ema[i]:
                signals.append(1)  # Buy signal
            elif short_ema[i] < long_ema[i]:
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