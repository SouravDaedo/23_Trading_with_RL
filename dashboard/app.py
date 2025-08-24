"""
Live Trading Dashboard for RL Agent Visualization
Real-time monitoring of agent actions, portfolio performance, and stock data
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import time
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardDataManager:
    """Manages real-time data for the dashboard"""
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.current_data = {
            'episode': 0,
            'step': 0,
            'portfolio_value': 10000,
            'current_price': 0,
            'actions': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'recent_actions': [],
            'portfolio_history': [],
            'price_history': [],
            'rewards': [],
            'agent_params': {
                'epsilon': 0, 
                'alpha': 0, 
                'loss': 0,
                'steps_per_second': 0,
                'episode_duration': 0
            },
            'performance_metrics': {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'episode_time': 0,
                'steps_per_second': 0
            },
            'positions': {'cash': 10000, 'shares': 0, 'total_value': 10000},
            'price_window': [],
            'timestamp_window': [],
            'episode_summary': {}
        }
        self.is_running = False
        
    def update_data(self, new_data):
        """Update dashboard data with new values from the training script"""
        # Update episode summary if provided
        if 'episode_summary' in new_data:
            self.current_data['episode_summary'] = new_data['episode_summary']
            
        # Update current data with new values
        for key, value in new_data.items():
            if key in ['actions', 'agent_params', 'positions']:
                # Handle nested dictionaries
                if key not in self.current_data:
                    self.current_data[key] = {}
                self.current_data[key].update(value)
                
                # Update performance metrics if available in agent_params
                if key == 'agent_params':
                    if 'steps_per_second' in value:
                        self.current_data['performance_metrics']['steps_per_second'] = value['steps_per_second']
                    if 'episode_duration' in value:
                        self.current_data['performance_metrics']['episode_time'] = value['episode_duration']
                        
            elif key == 'recent_actions':
                # Keep only the last 10 actions
                self.current_data['recent_actions'] = (self.current_data.get('recent_actions', []) + value)[-10:]
                
            elif key in ['portfolio_history', 'price_history', 'rewards']:
                # Append to history lists
                if key not in self.current_data:
                    self.current_data[key] = []
                if isinstance(value, (list, tuple)):
                    self.current_data[key].extend(value)
                else:
                    self.current_data[key].append(value)
                # Keep history lists at a reasonable size
                self.current_data[key] = self.current_data[key][-1000:]
                
            elif key == 'price_window':
                # Update price window data
                self.current_data['price_window'] = value
                # Also update the current price to the last value in the window
                if value and len(value) > 0:
                    self.current_data['current_price'] = value[-1]
                    
            elif key == 'timestamp_window':
                # Update timestamp window data
                self.current_data['timestamp_window'] = value
                
            elif key == 'episode_reward' and 'rewards' in self.current_data:
                # Handle episode reward specially to maintain rewards history
                self.current_data['rewards'].append(value)
                self.current_data['rewards'] = self.current_data['rewards'][-1000:]
                
            else:
                self.current_data[key] = value
                
        # Calculate performance metrics
        self.calculate_metrics()
    
    def get_current_data(self):
        """Get current dashboard data"""
        return self.current_data.copy()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.current_data['portfolio_history']) < 2:
            return
            
        portfolio_values = self.current_data['portfolio_history']
        initial_value = portfolio_values[0] if portfolio_values else 10000
        current_value = portfolio_values[-1] if portfolio_values else 10000
        
        # Total return
        total_return = ((current_value - initial_value) / initial_value) * 100
        
        # Calculate returns for Sharpe ratio
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
            
            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown) * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Win rate (based on positive rewards)
        rewards = self.current_data['rewards']
        win_rate = (np.sum(np.array(rewards) > 0) / len(rewards) * 100) if rewards else 0
        
        self.current_data['performance_metrics'] = {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 1)
        }

# Global data manager
data_manager = DashboardDataManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current data"""
    data_manager.calculate_metrics()
    return jsonify(data_manager.get_current_data())

@app.route('/api/update', methods=['POST'])
def update_data():
    """API endpoint to update data from training script"""
    try:
        new_data = request.json
        data_manager.update_data(new_data)
        
        # Emit to all connected clients
        socketio.emit('data_update', data_manager.get_current_data())
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('data_update', data_manager.get_current_data())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def simulate_data():
    """Simulate trading data for testing (remove in production)"""
    episode = 0
    step = 0
    portfolio = 10000
    price = 150
    
    while True:
        time.sleep(1)  # Update every second
        
        # Simulate price movement
        price += np.random.normal(0, 1)
        price = max(price, 50)  # Minimum price
        
        # Simulate agent action
        action = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
        
        # Simulate portfolio change
        if action == 'BUY':
            portfolio -= price * 10  # Buy 10 shares
        elif action == 'SELL':
            portfolio += price * 10  # Sell 10 shares
            
        reward = np.random.normal(0, 5)
        
        # Update data
        update_data = {
            'episode': episode,
            'step': step,
            'portfolio_value': portfolio,
            'current_price': price,
            'recent_actions': [{'action': action, 'price': price, 'time': datetime.now().strftime('%H:%M:%S')}],
            'portfolio_history': data_manager.current_data['portfolio_history'] + [portfolio],
            'price_history': data_manager.current_data['price_history'] + [price],
            'rewards': data_manager.current_data['rewards'] + [reward],
            'agent_params': {
                'epsilon': max(0.01, 0.5 - step * 0.001),
                'alpha': 0.3,
                'loss': abs(np.random.normal(0, 0.1))
            }
        }
        
        # Update action counts
        actions = data_manager.current_data['actions'].copy()
        actions[action] += 1
        update_data['actions'] = actions
        
        data_manager.update_data(update_data)
        socketio.emit('data_update', data_manager.get_current_data())
        
        step += 1
        if step % 100 == 0:
            episode += 1
            step = 0

if __name__ == '__main__':
    # Start simulation thread for testing
    # simulation_thread = threading.Thread(target=simulate_data, daemon=True)
    # simulation_thread.start()
    
    print("🚀 Starting Trading Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    
    # Suppress all logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    logging.getLogger('socketio').setLevel(logging.ERROR)
    logging.getLogger('engineio').setLevel(logging.ERROR)
    
    # Disable Flask and Socket.IO logging
    socketio.run(app, 
                debug=False, 
                host='0.0.0.0', 
                port=5000, 
                log_output=False, 
                use_reloader=False,
                allow_unsafe_werkzeug=True)
