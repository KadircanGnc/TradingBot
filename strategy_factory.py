from strategies import MomentumStrategy, MovingAverageStrategy, RSIStrategy, BollingerBandsStrategy, EMACrossoverStrategy

class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_type):
        if strategy_type == "momentum":
            return MomentumStrategy()
        elif strategy_type == "moving_average":
            return MovingAverageStrategy()
        elif strategy_type == "rsi":
            return RSIStrategy()
        elif strategy_type == "bollinger_bands":
            return BollingerBandsStrategy()
        elif strategy_type == "ema_crossover":
            return EMACrossoverStrategy()
        else:
            raise ValueError("Unknown strategy type")