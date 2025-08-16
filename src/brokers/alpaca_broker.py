"""
Example broker integration with Alpaca API for real trading.
WARNING: This involves real money - use paper trading account first!
"""

import alpaca_trade_api as tradeapi
from typing import Dict, Optional
import os
from datetime import datetime

class AlpacaBroker:
    """Alpaca broker integration for real trading execution."""
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca broker connection.
        
        Args:
            paper_trading: If True, use paper trading account (recommended)
        """
        # Get API credentials from environment variables
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Set base URL based on paper trading flag
        if paper_trading:
            base_url = 'https://paper-api.alpaca.markets'
            print(" Using Alpaca PAPER TRADING account")
        else:
            base_url = 'https://api.alpaca.markets'
            print(" Using Alpaca LIVE TRADING account - REAL MONEY!")
        
        # Initialize API connection
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url,
            api_version='v2'
        )
        
        # Verify connection
        try:
            account = self.api.get_account()
            print(f" Connected to Alpaca - Account Status: {account.status}")
            print(f" Buying Power: ${float(account.buying_power):,.2f}")
        except Exception as e:
            print(f" Failed to connect to Alpaca: {e}")
            raise
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        account = self.api.get_account()
        return {
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'day_trade_count': int(account.day_trade_count),
            'status': account.status
        }
    
    def get_current_position(self, symbol: str) -> Dict:
        """Get current position for a symbol."""
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': symbol,
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'unrealized_pl': float(position.unrealized_pl),
                'side': position.side
            }
        except Exception:
            # No position exists
            return {
                'symbol': symbol,
                'qty': 0.0,
                'market_value': 0.0,
                'avg_entry_price': 0.0,
                'unrealized_pl': 0.0,
                'side': 'flat'
            }
    
    def place_market_order(self, symbol: str, qty: int, side: str) -> Dict:
        """
        Place a market order.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            qty: Quantity to trade
            side: 'buy' or 'sell'
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            print(f" Order placed: {side.upper()} {qty} shares of {symbol}")
            print(f"Order ID: {order.id}")
            
            return {
                'order_id': order.id,
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'status': order.status,
                'submitted_at': order.submitted_at
            }
            
        except Exception as e:
            print(f" Order failed: {e}")
            return {'error': str(e)}
    
    def execute_rl_action(self, symbol: str, action: int, 
                         position_size: float = 100) -> Dict:
        """
        Execute RL agent action through broker.
        
        Args:
            symbol: Trading symbol
            action: 0=hold, 1=buy, 2=sell
            position_size: Number of shares to trade
        """
        current_pos = self.get_current_position(symbol)
        current_qty = current_pos['qty']
        
        if action == 0:  # Hold
            print(f" HOLD - No action for {symbol}")
            return {'action': 'hold', 'symbol': symbol}
        
        elif action == 1:  # Buy
            if current_qty >= 0:  # Not short, can buy more
                return self.place_market_order(symbol, int(position_size), 'buy')
            else:  # Currently short, close short position first
                return self.place_market_order(symbol, abs(int(current_qty)), 'buy')
        
        elif action == 2:  # Sell
            if current_qty > 0:  # Currently long, sell
                return self.place_market_order(symbol, int(current_qty), 'sell')
            elif current_qty == 0:  # No position, go short
                return self.place_market_order(symbol, int(position_size), 'sell')
            else:  # Already short
                print(f"⚠️  Already short {symbol}, no additional sell")
                return {'action': 'hold', 'symbol': symbol}
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get latest market data for symbol."""
        try:
            latest_trade = self.api.get_latest_trade(symbol)
            return {
                'symbol': symbol,
                'price': float(latest_trade.price),
                'timestamp': latest_trade.timestamp,
                'size': latest_trade.size
            }
        except Exception as e:
            print(f" Failed to get market data: {e}")
            return {'error': str(e)}
    
    def close_all_positions(self):
        """Close all open positions (emergency stop)."""
        try:
            self.api.close_all_positions()
            print(" All positions closed")
        except Exception as e:
            print(f"Failed to close positions: {e}")
    
    def get_order_history(self, limit: int = 10) -> list:
        """Get recent order history."""
        try:
            orders = self.api.list_orders(
                status='all',
                limit=limit,
                direction='desc'
            )
            
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_avg_price': order.filled_avg_price
            } for order in orders]
            
        except Exception as e:
            print(f" Failed to get order history: {e}")
            return []


# Example usage with RL agent
def integrate_with_rl_agent():
    """Example of how to integrate broker with RL agent."""
    
    # Initialize broker (PAPER TRADING!)
    broker = AlpacaBroker(paper_trading=True)
    
    # Get account info
    account = broker.get_account_info()
    print(f"Account Value: ${account['portfolio_value']:,.2f}")
    
    # Example: Execute RL agent decision
    symbol = "AAPL"
    rl_action = 1  # Buy signal from RL agent
    
    # Execute the trade
    result = broker.execute_rl_action(symbol, rl_action, position_size=10)
    print(f"Trade Result: {result}")
    
    # Check new position
    position = broker.get_current_position(symbol)
    print(f"New Position: {position}")


if __name__ == "__main__":
    # IMPORTANT: Set environment variables first!
    # export ALPACA_API_KEY="your_key_here"
    # export ALPACA_SECRET_KEY="your_secret_here"
    
    print(" BROKER INTEGRATION EXAMPLE")
    print("This is for educational purposes only!")
    print("Always test with paper trading first!")
    
    # Uncomment to test (after setting API keys)
    # integrate_with_rl_agent()
