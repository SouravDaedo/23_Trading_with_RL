import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class TradingMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
        """Calculate returns from portfolio values."""
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    
    @staticmethod
    def total_return(initial_value: float, final_value: float) -> float:
        """Calculate total return percentage."""
        return (final_value - initial_value) / initial_value * 100
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown) * 100
    
    @staticmethod
    def volatility(returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(252) * 100
    
    @staticmethod
    def win_rate(trade_history: List[Tuple]) -> float:
        """Calculate win rate from trade history."""
        if len(trade_history) < 2:
            return 0.0
        
        profitable_trades = 0
        total_trades = 0
        
        buy_price = None
        for trade in trade_history:
            action, shares, price, step = trade
            
            if action == 'BUY':
                buy_price = price
            elif action == 'SELL' and buy_price is not None:
                if price > buy_price:
                    profitable_trades += 1
                total_trades += 1
                buy_price = None
        
        return (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    @staticmethod
    def calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        return total_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_all_metrics(portfolio_values: List[float], 
                            trade_history: List[Tuple],
                            initial_value: float) -> Dict[str, float]:
        """Calculate comprehensive trading metrics."""
        if len(portfolio_values) == 0:
            return {}
        
        final_value = portfolio_values[-1]
        returns = TradingMetrics.calculate_returns(portfolio_values)
        
        metrics = {
            'total_return': TradingMetrics.total_return(initial_value, final_value),
            'sharpe_ratio': TradingMetrics.sharpe_ratio(returns),
            'max_drawdown': TradingMetrics.max_drawdown(portfolio_values),
            'volatility': TradingMetrics.volatility(returns),
            'win_rate': TradingMetrics.win_rate(trade_history),
            'final_portfolio_value': final_value,
            'total_trades': len(trade_history)
        }
        
        # Add Calmar ratio
        metrics['calmar_ratio'] = TradingMetrics.calmar_ratio(
            metrics['total_return'], 
            metrics['max_drawdown']
        )
        
        return metrics
