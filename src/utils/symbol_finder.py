"""
Stock Symbol Finder - Get available symbols for trading
"""

import yfinance as yf
import pandas as pd
import requests
from typing import List, Dict, Optional
import time

class SymbolFinder:
    """Find and validate stock symbols for trading."""
    
    def __init__(self):
        self.popular_stocks = self._get_popular_stocks()
        self.sector_stocks = self._get_sector_stocks()
    
    def _get_popular_stocks(self) -> Dict[str, List[str]]:
        """Get popular stock symbols by category."""
        return {
            "FAANG": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
            "Magnificent_7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"],
            "Dow_Jones_Top": ["AAPL", "MSFT", "UNH", "GS", "HD", "CAT", "CRM", "MCD", "V", "AXP"],
            "S&P_500_Top": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "META", "BRK-B", "UNH", "JNJ"],
            "Tech_Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "INTC"],
            "Banking": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "KMI", "OKE", "WMB", "MPC"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABT", "TMO", "AZN", "MRK", "DHR", "BMY", "AMGN"],
            "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "WMT", "PG", "KO"],
            "Finance": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "AXP", "GS", "MS", "SPGI"],
            "Crypto_Related": ["COIN", "MSTR", "RIOT", "MARA", "SQ", "PYPL", "HOOD", "SOFI"],
            "Meme_Stocks": ["GME", "AMC", "BB", "NOK", "PLTR", "WISH", "CLOV", "SPCE"],
            "EV_Stocks": ["TSLA", "NIO", "XPEV", "LI", "RIVN", "LCID", "F", "GM", "NKLA"],
            "AI_Stocks": ["NVDA", "AMD", "GOOGL", "MSFT", "META", "AMZN", "PLTR", "C3AI", "AI"],
            "Dividend_Kings": ["KO", "PEP", "JNJ", "PG", "MMM", "CAT", "XOM", "CVX", "IBM", "GE"]
        }
    
    def _get_sector_stocks(self) -> Dict[str, List[str]]:
        """Get stocks by sector."""
        return {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "INTC", "CSCO", "IBM", "QCOM", "TXN", "AVGO"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABT", "TMO", "AZN", "MRK", "DHR", "BMY", "AMGN", "GILD", "MDT", "CVS", "CI", "HUM"],
            "Financial": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "AXP", "GS", "MS", "SPGI", "BLK", "C", "USB", "PNC", "TFC"],
            "Consumer_Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "CMG", "ORLY", "YUM", "RCL"],
            "Communication": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "VZ", "T", "CHTR", "TMUS", "DISH", "PARA", "WBD", "FOXA", "IPG"],
            "Industrial": ["CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "FDX", "NOC", "GD", "EMR", "ETN", "ITW", "CSX"],
            "Consumer_Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KMB", "GIS", "K", "HSY", "MKC", "CPB", "CAG"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "KMI", "OKE", "WMB", "MPC", "VLO", "PSX", "HES", "DVN", "FANG"],
            "Utilities": ["NEE", "SO", "DUK", "AEP", "EXC", "XEL", "PEG", "SRE", "D", "PCG", "EIX", "WEC", "AWK", "ES", "FE"],
            "Real_Estate": ["PLD", "AMT", "CCI", "EQIX", "PSA", "EQR", "WELL", "DLR", "BXP", "VTR", "SBAC", "IRM", "ARE", "UDR"],
            "Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "DOW", "DD", "PPG", "ECL", "MLM", "VMC", "NUE", "STLD", "PKG", "IP"]
        }
    
    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols from all categories."""
        all_symbols = set()
        
        # Add popular stocks
        for category, symbols in self.popular_stocks.items():
            all_symbols.update(symbols)
        
        # Add sector stocks
        for sector, symbols in self.sector_stocks.items():
            all_symbols.update(symbols)
        
        return sorted(list(all_symbols))
    
    def get_symbols_by_category(self, category: str) -> List[str]:
        """Get symbols for a specific category."""
        if category in self.popular_stocks:
            return self.popular_stocks[category]
        elif category in self.sector_stocks:
            return self.sector_stocks[category]
        else:
            available = list(self.popular_stocks.keys()) + list(self.sector_stocks.keys())
            raise ValueError(f"Category '{category}' not found. Available: {available}")
    
    def validate_symbols(self, symbols: List[str], max_symbols: int = 10) -> Dict[str, Dict]:
        """
        Validate if symbols are tradeable and get basic info.
        
        Args:
            symbols: List of symbols to validate
            max_symbols: Maximum number of symbols to validate (to avoid rate limits)
        """
        results = {}
        symbols_to_check = symbols[:max_symbols]
        
        print(f"Validating {len(symbols_to_check)} symbols...")
        
        for i, symbol in enumerate(symbols_to_check):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get recent price data to verify it's tradeable
                hist = ticker.history(period="5d")
                
                if not hist.empty and info:
                    results[symbol] = {
                        'valid': True,
                        'name': info.get('longName', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A'),
                        'market_cap': info.get('marketCap', 'N/A'),
                        'current_price': hist['Close'].iloc[-1] if not hist.empty else 'N/A',
                        'currency': info.get('currency', 'USD'),
                        'exchange': info.get('exchange', 'N/A')
                    }
                else:
                    results[symbol] = {'valid': False, 'error': 'No data available'}
                
                # Rate limiting
                if i < len(symbols_to_check) - 1:
                    time.sleep(0.1)  # Small delay to avoid rate limits
                    
            except Exception as e:
                results[symbol] = {'valid': False, 'error': str(e)}
        
        return results
    
    def search_symbols(self, query: str) -> List[str]:
        """Search for symbols containing the query string."""
        query = query.upper()
        all_symbols = self.get_all_symbols()
        
        # Direct matches
        direct_matches = [s for s in all_symbols if query in s]
        
        return direct_matches
    
    def get_trading_recommendations(self, risk_level: str = "medium") -> Dict[str, List[str]]:
        """
        Get symbol recommendations based on risk level.
        
        Args:
            risk_level: "low", "medium", "high"
        """
        recommendations = {
            "low": {
                "description": "Large-cap, stable companies with consistent performance",
                "symbols": ["AAPL", "MSFT", "JNJ", "PG", "KO", "WMT", "V", "MA", "UNH", "HD"]
            },
            "medium": {
                "description": "Mix of large-cap growth and established companies",
                "symbols": ["GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "CRM", "ADBE", "PYPL", "COST"]
            },
            "high": {
                "description": "Growth stocks, volatile sectors, and emerging companies",
                "symbols": ["PLTR", "COIN", "RIVN", "LCID", "SPCE", "GME", "AMC", "MARA", "RIOT", "AI"]
            }
        }
        
        if risk_level not in recommendations:
            raise ValueError("Risk level must be 'low', 'medium', or 'high'")
        
        return recommendations[risk_level]
    
    def print_all_categories(self):
        """Print all available categories and their symbols."""
        print(" POPULAR STOCK CATEGORIES:")
        print("=" * 50)
        for category, symbols in self.popular_stocks.items():
            print(f"\n {category.replace('_', ' ').title()}:")
            print(f"   {', '.join(symbols)}")
        
        print("\n\n SECTOR CATEGORIES:")
        print("=" * 50)
        for sector, symbols in self.sector_stocks.items():
            print(f"\n🔹 {sector.replace('_', ' ').title()}:")
            print(f"   {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
            print(f"   ({len(symbols)} total symbols)")


def main():
    """Example usage of SymbolFinder."""
    finder = SymbolFinder()
    
    print(" STOCK SYMBOL FINDER")
    print("=" * 50)
    
    # Show all categories
    finder.print_all_categories()
    
    # Get all unique symbols
    all_symbols = finder.get_all_symbols()
    print(f"\n TOTAL UNIQUE SYMBOLS: {len(all_symbols)}")
    
    # Show some examples
    print(f"\nFirst 20 symbols: {', '.join(all_symbols[:20])}")
    
    # Validate a few popular symbols
    print(f"\n🔍 VALIDATING POPULAR SYMBOLS:")
    popular_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    validation_results = finder.validate_symbols(popular_symbols)
    
    for symbol, result in validation_results.items():
        if result['valid']:
            print(f" {symbol}: {result['name']} - ${result['current_price']:.2f}")
        else:
            print(f" {symbol}: {result['error']}")
    
    # Show recommendations
    print(f"\n TRADING RECOMMENDATIONS:")
    for risk in ["low", "medium", "high"]:
        rec = finder.get_trading_recommendations(risk)
        print(f"\n{risk.upper()} RISK: {rec['description']}")
        print(f"   {', '.join(rec['symbols'])}")


if __name__ == "__main__":
    main()
