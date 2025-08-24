"""
Price Data Viewer - Export and inspect stock price data used in training
"""

import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append('src')
from data.data_loader import DataLoader

def export_price_data_to_csv(config_path="config/config.yaml", output_dir="data/"):
    """Export price data for all configured symbols to CSV files."""
    
    print("Loading configuration and downloading price data...")
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize data loader
    data_loader = DataLoader(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for symbol in config['data']['symbols']:
        print(f"\nProcessing {symbol}...")
        
        # Download raw data
        raw_data = data_loader.download_data(symbol)
        
        if raw_data.empty:
            print(f"No data found for {symbol}")
            continue
            
        # Add technical indicators
        processed_data = data_loader.add_technical_indicators(raw_data)
        
        # Prepare data for training (normalize)
        train_data, test_data = data_loader.prepare_data(symbol)
        
        # Export raw price data
        raw_csv_path = os.path.join(output_dir, f"{symbol}_raw_prices.csv")
        raw_data.to_csv(raw_csv_path)
        print(f"Raw price data saved: {raw_csv_path}")
        
        # Export processed data with indicators
        processed_csv_path = os.path.join(output_dir, f"{symbol}_processed_data.csv")
        processed_data.to_csv(processed_csv_path)
        print(f"Processed data saved: {processed_csv_path}")
        
        # Export training data (normalized)
        train_csv_path = os.path.join(output_dir, f"{symbol}_train.csv")
        train_data.to_csv(train_csv_path)
        print(f"Training data saved: {train_csv_path}")
        
        # Export test data
        test_csv_path = os.path.join(output_dir, f"{symbol}_test.csv")
        test_data.to_csv(test_csv_path)
        print(f"Test data saved: {test_csv_path}")
        
        # Create price summary
        price_summary = {
            'symbol': symbol,
            'total_periods': len(raw_data),
            'date_range': f"{raw_data.index[0].strftime('%Y-%m-%d')} to {raw_data.index[-1].strftime('%Y-%m-%d')}",
            'price_range': f"${raw_data['Close'].min():.2f} - ${raw_data['Close'].max():.2f}",
            'latest_price': f"${raw_data['Close'].iloc[-1]:.2f}",
            'training_samples': len(train_data),
            'test_samples': len(test_data),
            'interval': config['data'].get('interval', '1d')
        }
        
        results[symbol] = price_summary
        
        print(f"Price Range: ${raw_data['Close'].min():.2f} - ${raw_data['Close'].max():.2f}")
        print(f"Date Range: {price_summary['date_range']}")
        print(f"Training Samples: {len(train_data)}, Test Samples: {len(test_data)}")
    
    # Create summary report
    summary_path = os.path.join(output_dir, "price_data_summary.csv")
    summary_df = pd.DataFrame.from_dict(results, orient='index')
    summary_df.to_csv(summary_path)
    print(f"\nSummary report saved: {summary_path}")
    
    return results

def create_price_visualization(config_path="config/config.yaml", output_dir="data/plots/"):
    """Create price visualization charts for all symbols."""
    
    print("\nCreating price visualizations...")
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    data_loader = DataLoader(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in config['data']['symbols']:
        print(f"Creating chart for {symbol}...")
        
        # Download and process data
        raw_data = data_loader.download_data(symbol)
        if raw_data.empty:
            continue
            
        processed_data = data_loader.add_technical_indicators(raw_data)
        
        # Create comprehensive price chart
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{symbol} Price Analysis', fontsize=16)
        
        # Price and moving averages
        axes[0].plot(processed_data.index, processed_data['Close'], label='Close Price', linewidth=1)
        if 'SMA_10' in processed_data.columns:
            axes[0].plot(processed_data.index, processed_data['SMA_10'], label='SMA 10', alpha=0.7)
        if 'SMA_20' in processed_data.columns:
            axes[0].plot(processed_data.index, processed_data['SMA_20'], label='SMA 20', alpha=0.7)
        axes[0].set_title('Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        if 'Volume' in processed_data.columns:
            axes[1].bar(processed_data.index, processed_data['Volume'], alpha=0.6, color='orange')
            axes[1].set_title('Trading Volume')
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in processed_data.columns:
            axes[2].plot(processed_data.index, processed_data['RSI'], color='purple')
            axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            axes[2].set_title('RSI (Relative Strength Index)')
            axes[2].set_ylabel('RSI')
            axes[2].set_xlabel('Date')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, f"{symbol}_price_analysis.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Chart saved: {chart_path}")

def show_current_prices(config_path="config/config.yaml"):
    """Show current/latest prices for all configured symbols."""
    
    print("Current Stock Prices:")
    print("=" * 50)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    data_loader = DataLoader(config_path)
    
    for symbol in config['data']['symbols']:
        try:
            data = data_loader.download_data(symbol)
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                latest_date = data.index[-1].strftime('%Y-%m-%d')
                print(f"{symbol:>6}: ${latest_price:>8.2f} (as of {latest_date})")
            else:
                print(f"{symbol:>6}: No data available")
        except Exception as e:
            print(f"{symbol:>6}: Error - {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View and export stock price data')
    parser.add_argument('--export', action='store_true', help='Export price data to CSV files')
    parser.add_argument('--charts', action='store_true', help='Create price visualization charts')
    parser.add_argument('--current', action='store_true', help='Show current prices')
    parser.add_argument('--all', action='store_true', help='Run all operations')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    if args.all or not any([args.export, args.charts, args.current]):
        # Run all operations by default
        print("Running complete price data analysis...")
        show_current_prices(args.config)
        results = export_price_data_to_csv(args.config)
        create_price_visualization(args.config)
        
        print("\n" + "=" * 60)
        print("COMPLETE ANALYSIS FINISHED")
        print("=" * 60)
        print("Check the 'data/' folder for CSV files")
        print("Check the 'data/plots/' folder for charts")
        
    else:
        if args.current:
            show_current_prices(args.config)
        
        if args.export:
            export_price_data_to_csv(args.config)
        
        if args.charts:
            create_price_visualization(args.config)
    
    print("\nPrice data analysis complete!")
