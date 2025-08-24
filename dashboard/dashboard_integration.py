"""
Dashboard Integration Module
Connects the training script with the live dashboard for real-time visualization
"""

import requests
import json
import time
from datetime import datetime
import threading
import queue

class DashboardConnector:
    """Connects training script to live dashboard"""
    
    def __init__(self, dashboard_url="http://localhost:5000"):
        self.dashboard_url = dashboard_url
        self.update_queue = queue.Queue()
        self.is_connected = False
        self.session = requests.Session()
        
        # Start background thread for sending updates
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()
        
    def _update_worker(self):
        """Background worker to send updates to dashboard"""
        while True:
            try:
                if not self.update_queue.empty():
                    data = self.update_queue.get()
                    self._send_update(data)
                time.sleep(0.1)  # Small delay to prevent overwhelming
            except Exception as e:
                print(f"Dashboard update error: {e}")
                
    def _send_update(self, data):
        """Send update to dashboard"""
        try:
            response = self.session.post(
                f"{self.dashboard_url}/api/update",
                json=data,
                timeout=1.0
            )
            if response.status_code == 200:
                self.is_connected = True
            else:
                self.is_connected = False
        except Exception as e:
            self.is_connected = False
            # Silently fail to avoid disrupting training
            
    def update_step(self, episode, step, action, reward, portfolio_value, 
                   current_price, agent_params, positions=None, price_window=None, timestamp_window=None):
        """Update dashboard with step data"""
        
        # Convert action to string format
        if hasattr(action, '__len__'):  # Array/tensor
            if len(action) == 1:
                action_val = float(action[0])
                if action_val < -0.1:
                    action_str = 'SELL'
                elif action_val > 0.1:
                    action_str = 'BUY'
                else:
                    action_str = 'HOLD'
            else:
                action_str = 'HOLD'
        elif isinstance(action, (int, float)):
            if action == 0:
                action_str = 'HOLD'
            elif action == 1:
                action_str = 'BUY'
            elif action == 2:
                action_str = 'SELL'
            else:
                action_str = 'HOLD'
        else:
            action_str = 'HOLD'
            
        # Create update data with all required fields
        update_data = {
            'episode': int(episode) if episode is not None else 0,
            'step': int(step) if step is not None else 0,
            'portfolio_value': float(portfolio_value) if portfolio_value is not None else 0.0,
            'current_price': float(current_price) if current_price is not None else 0.0,
            'recent_actions': [{
                'action': action_str,
                'price': float(current_price) if current_price is not None else 0.0,
                'time': datetime.now().strftime('%H:%M:%S'),
                'reward': float(reward) if reward is not None else 0.0
            }],
            'agent_params': {
                'epsilon': float(agent_params.get('epsilon', 0)),
                'alpha': float(agent_params.get('alpha', 0)),
                'loss': float(agent_params.get('loss', 0)),
                'steps_per_second': float(agent_params.get('steps_per_second', 0)),
                'episode_duration': float(agent_params.get('episode_duration', 0))
            },
            'actions': {
                'BUY': int(agent_params.get('buy_count', 0)),
                'SELL': int(agent_params.get('sell_count', 0)),
                'HOLD': int(agent_params.get('hold_count', 0))
            },
            'portfolio_history': [float(portfolio_value)] if portfolio_value is not None else [0.0],
            'price_history': [float(current_price)] if current_price is not None else [0.0],
            'rewards': [float(reward)] if reward is not None else [0.0],
            'price_window': [float(p) for p in (price_window or [current_price])] if current_price is not None else [0.0],
            'timestamp_window': timestamp_window or [str(i) for i in range(len(price_window or [current_price] if current_price is not None else [0.0]))],
            'positions': positions or {'cash': 0, 'shares': 0, 'total_value': portfolio_value or 0},
            'episode_summary': {
                'reward': float(reward) if reward is not None else 0.0,
                'portfolio_value': float(portfolio_value) if portfolio_value is not None else 0.0,
                'actions': {
                    'BUY': int(agent_params.get('buy_count', 0)),
                    'SELL': int(agent_params.get('sell_count', 0)),
                    'HOLD': int(agent_params.get('hold_count', 0))
                }
            }
        }
        
        if positions:
            update_data['positions'] = {
                'cash': float(positions.get('cash', 0)),
                'shares': int(positions.get('shares', 0)),
                'total_value': float(positions.get('total_value', portfolio_value))
            }
            
        # Add to queue for background sending
        if not self.update_queue.full():
            self.update_queue.put(update_data)
            
    def update_episode_end(self, episode, episode_reward, portfolio_history, 
                          price_history, rewards_history, actions_summary):
        """Update dashboard at episode end"""
        
        update_data = {
            'episode': episode,
            'episode_reward': float(episode_reward),
            'portfolio_history': [float(x) for x in portfolio_history[-100:]],  # Last 100 points
            'price_history': [float(x) for x in price_history[-100:]],
            'rewards': [float(x) for x in rewards_history[-100:]],
            'actions': {
                'BUY': int(actions_summary.get('BUY', 0)),
                'SELL': int(actions_summary.get('SELL', 0)),
                'HOLD': int(actions_summary.get('HOLD', 0))
            }
        }
        
        if not self.update_queue.full():
            self.update_queue.put(update_data)
            
    def is_dashboard_connected(self):
        """Check if dashboard is connected"""
        return self.is_connected


class DashboardIntegratedMonitor:
    """Enhanced progress monitor with dashboard integration"""
    
    def __init__(self, total_episodes: int, agent_type: str = 'sac', 
                 enable_dashboard: bool = True):
        self.total_episodes = total_episodes
        self.agent_type = agent_type
        self.enable_dashboard = enable_dashboard
        
        # Dashboard connector
        self.dashboard = DashboardConnector() if enable_dashboard else None
        
        # Tracking variables
        self.current_episode = 0
        self.current_step = 0
        self.current_reward = 0
        self.current_portfolio = 10000
        self.current_price = 0
        self.current_epsilon = 0
        self.current_loss = 0
        
        # History tracking
        self.portfolio_history = []
        self.price_history = []
        self.price_timestamps = []  # Track timestamps for price data
        self.rewards_history = []
        self.episode_actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        self.total_actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Current price tracking
        self.current_price = 0
        self.trading_day = 0  # Track trading days for timeline
        
        # Performance tracking
        self.episode_start_time = 0
        self.steps_per_second = 0
        
    def start_episode(self, episode_num: int):
        """Start tracking a new episode"""
        self.current_episode = episode_num
        self.current_step = 0
        self.current_reward = 0
        self.episode_start_time = time.time()
        self.episode_actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
    def update_step(self, action, reward: float, portfolio_value: float,
                   current_price: float, epsilon_or_alpha: float = None, 
                   loss: float = None, positions: dict = None):
        """Update step-level metrics and send to dashboard"""
        
        self.current_step += 1
        self.current_reward += reward
        self.current_portfolio = portfolio_value
        self.current_price = current_price
        
        # Store current price and timestamp for price history
        if current_price > 0:
            self.price_history.append(current_price)
            # Create timeline based on trading days (since we're using daily data)
            self.price_timestamps.append(self.trading_day)
            self.trading_day += 1
        
        # Convert epsilon/alpha to scalar
        if epsilon_or_alpha is not None:
            if hasattr(epsilon_or_alpha, 'item'):
                self.current_epsilon = epsilon_or_alpha.item()
            else:
                self.current_epsilon = float(epsilon_or_alpha)
        
        # Convert loss to scalar
        if loss is not None:
            if hasattr(loss, 'item'):
                self.current_loss = loss.item()
            elif isinstance(loss, dict):
                self.current_loss = loss.get('critic1_loss', 0.0)
                if hasattr(self.current_loss, 'item'):
                    self.current_loss = self.current_loss.item()
            else:
                self.current_loss = float(loss)
        
        # Track actions
        if self.agent_type.lower() == 'dqn':
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_name = action_names.get(int(action), 'HOLD')
        else:  # SAC continuous actions
            action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
            if action_val < -0.1:
                action_name = 'SELL'
            elif action_val > 0.1:
                action_name = 'BUY'
            else:
                action_name = 'HOLD'
                
        self.episode_actions[action_name] += 1
        self.total_actions[action_name] += 1
        
        # Update current values
        self.current_portfolio = portfolio_value
        self.current_reward = reward
        self.current_price = current_price
        
        # Update histories
        self.portfolio_history.append(portfolio_value)
        self.price_history.append(current_price)
        self.rewards_history.append(reward)
        
        # Calculate performance
        if self.current_step > 0:
            episode_duration = time.time() - self.episode_start_time
            self.steps_per_second = self.current_step / max(episode_duration, 0.001)
        
        # Send to dashboard
        if self.dashboard:
            agent_params = {
                'epsilon' if self.agent_type.lower() == 'dqn' else 'alpha': self.current_epsilon,
                'loss': self.current_loss
            }
            
            # Prepare 5-day window data for dashboard
            price_history = self.price_history
            
            # If we have less than 5 data points, pad with the first available price
            if len(price_history) < 5:
                price_history = [price_history[0]] * (5 - len(price_history)) + price_history
            
            # Take the last 5 prices
            price_window = price_history[-5:]
            
            # Generate day labels (D-4, D-3, D-2, D-1, Current)
            days_ago = min(4, len(price_window) - 1)
            timestamp_window = [f'D-{days_ago - i}' for i in range(days_ago, -1, -1)]
            
            self.dashboard.update_step(
                episode=self.current_episode,
                step=self.current_step,
                action=action,
                reward=reward,
                portfolio_value=portfolio_value,
                current_price=current_price,
                agent_params=agent_params,
                positions=positions,
                price_window=price_window,
                timestamp_window=timestamp_window
            )
    
    def finish_episode(self):
        """Finish episode and send summary to dashboard if enabled"""
        # Calculate episode duration and steps per second
        episode_duration = time.time() - self.episode_start_time
        steps_per_second = self.current_step / max(episode_duration, 0.001)
        
        # Print episode summary to console
        print(f"\nEpisode {self.current_episode + 1} completed in {episode_duration:.2f}s "
              f"({steps_per_second:.1f} steps/sec)")
        print(f"  Total reward: {self.current_reward:.2f}")
        print(f"  Final portfolio value: {self.current_portfolio:.2f}")
        print(f"  Actions taken: {self.episode_actions}")
        
        # Send episode summary to dashboard if enabled
        if self.dashboard:
            self.dashboard.update_episode_end(
                episode=self.current_episode,
                episode_reward=self.current_reward,
                portfolio_history=self.portfolio_history,
                price_history=self.price_history,
                rewards_history=self.rewards_history,
                actions_summary=self.episode_actions  # Use episode actions instead of total
            )
            
            # Add performance metrics to dashboard
            self.dashboard.update_step(
                episode=self.current_episode,
                step=self.current_step,
                action=None,
                reward=self.current_reward,
                portfolio_value=self.current_portfolio,
                current_price=self.current_price,
                agent_params={
                    'epsilon' if self.agent_type.lower() == 'dqn' else 'alpha': self.current_epsilon,
                    'loss': self.current_loss,
                    'steps_per_second': steps_per_second,
                    'episode_duration': episode_duration
                },
                positions={
                    'cash': self.current_portfolio,
                    'shares': 0,  # This would need to be tracked in the environment
                    'total_value': self.current_portfolio
                }
            )
    
    def print_live_progress(self):
        """Print live progress (same as original but with dashboard status)"""
        progress_pct = (self.current_episode / self.total_episodes) * 100
        
        # Progress bar
        bar_length = 50
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Clear screen and print header
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print("=" * 80)
        print(f" LIVE TRAINING PROGRESS - Episode {self.current_episode}/{self.total_episodes}")
        if self.dashboard and self.dashboard.is_dashboard_connected():
            print(" 📊 Dashboard: Connected at http://localhost:5000")
        elif self.dashboard:
            print(" 📊 Dashboard: Disconnected")
        print("=" * 80)
        print(f"Progress: [{bar}] {progress_pct:.1f}%")
        
        # Current episode stats
        print(f"\n Current Episode:")
        print(f"  Step: {self.current_step:4d} | Reward: {self.current_reward:8.2f} | Portfolio: ${self.current_portfolio:8.0f}")
        print(f"  Actions: S:{self.episode_actions['SELL']:3d} H:{self.episode_actions['HOLD']:3d} B:{self.episode_actions['BUY']:3d}")
        param_name = "Alpha" if self.agent_type == 'sac' else "Epsilon"
        print(f"  {param_name}: {self.current_epsilon:.3f} | Loss: {self.current_loss:.4f} | Speed: {self.steps_per_second:.1f} steps/s")
        
        # Total actions
        print(f"\n Total Actions:")
        print(f"  BUY: {self.total_actions['BUY']:4d} | HOLD: {self.total_actions['HOLD']:4d} | SELL: {self.total_actions['SELL']:4d}")
        
        print("\n" + "=" * 80)
