import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """
    A class to backtest trading strategies based on ML signals.
    """
    
    def __init__(self, signals_path='outputs/predictions/signals.csv',
                 data_dir='data/processed',
                 outputs_dir='outputs',
                 initial_capital=10000,
                 position_size=0.1):
        """
        Initialize the Backtester.
        
        Parameters:
        -----------
        signals_path : str
            Path to signals CSV file
        data_dir : str
            Directory containing processed data
        outputs_dir : str
            Directory to save outputs
        initial_capital : float
            Starting capital in dollars
        position_size : float
            Fraction of capital to allocate per position (default: 10%)
        """
        self.signals_path = signals_path
        self.data_dir = data_dir
        self.outputs_dir = outputs_dir
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.signals_df = None
        self.backtest_results = {}
        
        # Create output directories
        Path(f"{outputs_dir}/backtest_results").mkdir(parents=True, exist_ok=True)
        
    def load_signals(self):
        """Load trading signals from CSV."""
        print("Loading trading signals...")
        
        self.signals_df = pd.read_csv(self.signals_path)
        self.signals_df['date'] = pd.to_datetime(self.signals_df['date'])
        
        print(f"Loaded {len(self.signals_df)} signals")
        print(f"Date range: {self.signals_df['date'].min()} to {self.signals_df['date'].max()}")
        print(f"Tickers: {self.signals_df['ticker'].unique()}")
        
        return self.signals_df
    
    def load_price_data(self):
        """Load price data for backtesting."""
        print("\nLoading price data...")
        
        # Load test data which has actual prices
        test_df = pd.read_csv(f"{self.data_dir}/test_data.csv")
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        # Keep only date, ticker, close price
        price_df = test_df[['date', 'ticker', 'close']].copy()
        
        print(f"Loaded price data for {len(price_df)} observations")
        
        return price_df
    
    def run_backtest(self, signal_col='ensemble_signal', model_name='ensemble'):
        """
        Run backtest simulation for a specific signal column.
        
        Parameters:
        -----------
        signal_col : str
            Column name containing signals (-1, 0, 1)
        model_name : str
            Name of the model/strategy
            
        Returns:
        --------
        dict
            Backtest results including equity curve and metrics
        """
        print(f"\n{'='*60}")
        print(f"Running backtest for: {model_name}")
        print(f"{'='*60}")
        
        # Merge signals with price data
        price_df = self.load_price_data()
        df = self.signals_df.merge(
            price_df, 
            on=['date', 'ticker'], 
            how='left'
        )
        
        # Sort by date and ticker
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        # Initialize tracking variables
        cash = self.initial_capital
        positions = {}  # {ticker: {'shares': n, 'entry_price': p}}
        portfolio_value = []
        dates = []
        trades = []
        
        # Get unique dates
        unique_dates = sorted(df['date'].unique())
        
        print(f"Simulating {len(unique_dates)} trading days...")
        
        for date in unique_dates:
            day_data = df[df['date'] == date]
            
            # Process signals for this day
            for _, row in day_data.iterrows():
                ticker = row['ticker']
                signal = row[signal_col]
                price = row['close']
                
                # Skip if price is NaN
                if pd.isna(price):
                    continue
                
                # Execute trades based on signals
                if signal == 1:  # BUY signal
                    if ticker not in positions and cash > 0:
                        # Calculate position size
                        position_value = cash * self.position_size
                        shares = position_value / price
                        cost = shares * price
                        
                        if cost <= cash:
                            # Open position
                            positions[ticker] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': date
                            }
                            cash -= cost
                            
                            trades.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'value': cost
                            })
                
                elif signal == -1:  # SELL signal
                    if ticker in positions:
                        # Close position
                        position = positions[ticker]
                        shares = position['shares']
                        proceeds = shares * price
                        cash += proceeds
                        
                        # Calculate P&L
                        entry_price = position['entry_price']
                        pnl = (price - entry_price) * shares
                        pnl_pct = (price / entry_price - 1) * 100
                        
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'SELL',
                            'price': price,
                            'shares': shares,
                            'value': proceeds,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'entry_price': entry_price,
                            'entry_date': position['entry_date']
                        })
                        
                        del positions[ticker]
                
                # HOLD (signal == 0): do nothing
            
            # Calculate portfolio value at end of day
            positions_value = sum(
                pos['shares'] * day_data[day_data['ticker'] == ticker]['close'].iloc[0]
                for ticker, pos in positions.items()
                if ticker in day_data['ticker'].values
            )
            total_value = cash + positions_value
            
            portfolio_value.append(total_value)
            dates.append(date)
        
        # Create equity curve dataframe
        equity_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_value
        })
        
        # Create trades dataframe
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate metrics
        metrics = self.calculate_metrics(equity_df, trades_df)
        
        # Store results
        results = {
            'model_name': model_name,
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'final_value': portfolio_value[-1] if portfolio_value else self.initial_capital,
            'total_return': (portfolio_value[-1] / self.initial_capital - 1) * 100 if portfolio_value else 0
        }
        
        self.backtest_results[model_name] = results
        
        # Print summary
        self.print_backtest_summary(model_name, metrics)
        
        return results
    
    def calculate_metrics(self, equity_df, trades_df):
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        equity_df : pd.DataFrame
            Equity curve dataframe
        trades_df : pd.DataFrame
            Trades dataframe
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Total return
        initial_value = equity_df['portfolio_value'].iloc[0]
        final_value = equity_df['portfolio_value'].iloc[-1]
        metrics['total_return'] = (final_value / initial_value - 1) * 100
        
        # Calculate daily returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        
        # Annualized return (assuming 252 trading days)
        num_days = len(equity_df)
        if num_days > 0:
            total_return_decimal = final_value / initial_value - 1
            metrics['annualized_return'] = ((1 + total_return_decimal) ** (252 / num_days) - 1) * 100
        else:
            metrics['annualized_return'] = 0
        
        # Volatility (annualized)
        metrics['volatility'] = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] / equity_df['cummax'] - 1) * 100
        metrics['max_drawdown'] = equity_df['drawdown'].min()
        
        # Trade statistics
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            closed_trades = trades_df[trades_df['action'] == 'SELL']
            
            if len(closed_trades) > 0:
                metrics['num_trades'] = len(closed_trades)
                metrics['win_rate'] = (closed_trades['pnl'] > 0).sum() / len(closed_trades) * 100
                metrics['avg_gain'] = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean()
                metrics['avg_loss'] = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean()
                metrics['avg_gain_pct'] = closed_trades[closed_trades['pnl_pct'] > 0]['pnl_pct'].mean()
                metrics['avg_loss_pct'] = closed_trades[closed_trades['pnl_pct'] < 0]['pnl_pct'].mean()
                
                # Profit factor
                total_gains = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
                total_losses = abs(closed_trades[closed_trades['pnl'] < 0]['pnl'].sum())
                metrics['profit_factor'] = total_gains / total_losses if total_losses > 0 else np.inf
            else:
                metrics['num_trades'] = 0
                metrics['win_rate'] = 0
                metrics['avg_gain'] = 0
                metrics['avg_loss'] = 0
                metrics['avg_gain_pct'] = 0
                metrics['avg_loss_pct'] = 0
                metrics['profit_factor'] = 0
        else:
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0
            metrics['avg_gain'] = 0
            metrics['avg_loss'] = 0
            metrics['avg_gain_pct'] = 0
            metrics['avg_loss_pct'] = 0
            metrics['profit_factor'] = 0
        
        return metrics
    
    def run_buy_and_hold_benchmark(self):
        """
        Run buy-and-hold benchmark for comparison.
        
        Returns:
        --------
        dict
            Benchmark results
        """
        print(f"\n{'='*60}")
        print("Running Buy-and-Hold Benchmark")
        print(f"{'='*60}")
        
        price_df = self.load_price_data()
        
        # Get unique dates and tickers
        unique_dates = sorted(price_df['date'].unique())
        tickers = price_df['ticker'].unique()
        
        # Equal weight allocation across all tickers
        capital_per_ticker = self.initial_capital / len(tickers)
        
        # Buy at first date
        first_date = unique_dates[0]
        positions = {}
        
        for ticker in tickers:
            first_price = price_df[(price_df['date'] == first_date) & 
                                   (price_df['ticker'] == ticker)]['close'].iloc[0]
            shares = capital_per_ticker / first_price
            positions[ticker] = shares
        
        # Calculate portfolio value over time
        portfolio_values = []
        dates = []
        
        for date in unique_dates:
            day_prices = price_df[price_df['date'] == date]
            
            total_value = 0
            for ticker, shares in positions.items():
                if ticker in day_prices['ticker'].values:
                    price = day_prices[day_prices['ticker'] == ticker]['close'].iloc[0]
                    total_value += shares * price
            
            portfolio_values.append(total_value)
            dates.append(date)
        
        # Create equity curve
        equity_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        # Calculate metrics
        metrics = self.calculate_metrics(equity_df, pd.DataFrame())
        
        results = {
            'model_name': 'Buy-and-Hold',
            'equity_curve': equity_df,
            'trades': pd.DataFrame(),
            'metrics': metrics,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / self.initial_capital - 1) * 100
        }
        
        self.backtest_results['Buy-and-Hold'] = results
        
        # Print summary
        self.print_backtest_summary('Buy-and-Hold', metrics)
        
        return results
    
    def print_backtest_summary(self, model_name, metrics):
        """Print formatted backtest summary."""
        print(f"\n{model_name} Performance:")
        print("-" * 60)
        print(f"Total Return:        {metrics['total_return']:>10.2f}%")
        print(f"Annualized Return:   {metrics['annualized_return']:>10.2f}%")
        print(f"Volatility:          {metrics['volatility']:>10.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2f}%")
        print(f"Number of Trades:    {metrics['num_trades']:>10.0f}")
        
        if metrics['num_trades'] > 0:
            print(f"Win Rate:            {metrics['win_rate']:>10.2f}%")
            print(f"Avg Gain:            ${metrics['avg_gain']:>10.2f} ({metrics['avg_gain_pct']:.2f}%)")
            print(f"Avg Loss:            ${metrics['avg_loss']:>10.2f} ({metrics['avg_loss_pct']:.2f}%)")
            print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
    
    def run_all_backtests(self):
        """Run backtests for all available signals."""
        print("\n" + "="*60)
        print("RUNNING ALL BACKTESTS")
        print("="*60)
        
        # Get all signal columns
        signal_cols = [col for col in self.signals_df.columns if col.endswith('_signal')]
        
        # Run backtest for each signal
        for signal_col in signal_cols:
            model_name = signal_col.replace('_signal', '')
            self.run_backtest(signal_col, model_name)
        
        # Run buy-and-hold benchmark
        self.run_buy_and_hold_benchmark()
        
        return self.backtest_results
    
    def plot_equity_curves(self):
        """Plot equity curves for all strategies."""
        print("\nGenerating equity curve plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for model_name, results in self.backtest_results.items():
            equity_df = results['equity_curve']
            
            # Plot with different style for benchmark
            if model_name == 'Buy-and-Hold':
                ax.plot(equity_df['date'], equity_df['portfolio_value'], 
                       label=model_name, linewidth=2.5, linestyle='--', alpha=0.7)
            else:
                ax.plot(equity_df['date'], equity_df['portfolio_value'], 
                       label=model_name, linewidth=2, alpha=0.8)
        
        # Add initial capital line
        ax.axhline(y=self.initial_capital, color='gray', linestyle=':', 
                  linewidth=1, label='Initial Capital', alpha=0.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Equity Curves - Strategy Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        save_path = f"{self.outputs_dir}/backtest_results/equity_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved equity curves: {save_path}")
        plt.close()
    
    def plot_drawdowns(self):
        """Plot drawdown charts for all strategies."""
        print("Generating drawdown plot...")
        
        n_strategies = len(self.backtest_results)
        fig, axes = plt.subplots(n_strategies, 1, figsize=(14, 4*n_strategies))
        
        if n_strategies == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.backtest_results.items()):
            equity_df = results['equity_curve'].copy()
            
            # Calculate drawdown
            equity_df['cummax'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] / equity_df['cummax'] - 1) * 100
            
            axes[idx].fill_between(equity_df['date'], equity_df['drawdown'], 0, 
                                  alpha=0.3, color='red')
            axes[idx].plot(equity_df['date'], equity_df['drawdown'], 
                          color='red', linewidth=1.5)
            axes[idx].set_ylabel('Drawdown (%)', fontsize=10)
            axes[idx].set_title(f'{model_name} - Drawdown', fontsize=11)
            axes[idx].grid(alpha=0.3)
            axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        save_path = f"{self.outputs_dir}/backtest_results/drawdowns.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved drawdown plot: {save_path}")
        plt.close()
    
    def plot_returns_distribution(self):
        """Plot distribution of daily returns."""
        print("Generating returns distribution plot...")
        
        fig, axes = plt.subplots(1, len(self.backtest_results), 
                                figsize=(5*len(self.backtest_results), 4))
        
        if len(self.backtest_results) == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.backtest_results.items()):
            equity_df = results['equity_curve'].copy()
            equity_df['returns'] = equity_df['portfolio_value'].pct_change() * 100
            daily_returns = equity_df['returns'].dropna()
            
            axes[idx].hist(daily_returns, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[idx].set_xlabel('Daily Return (%)', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{model_name}\nReturns Distribution', fontsize=11)
            axes[idx].grid(alpha=0.3)
            
            # Add statistics
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            axes[idx].text(0.02, 0.98, 
                          f'Mean: {mean_return:.3f}%\nStd: {std_return:.3f}%',
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = f"{self.outputs_dir}/backtest_results/returns_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved returns distribution: {save_path}")
        plt.close()
    
    def create_comparison_table(self):
        """Create comparison table of all strategies."""
        print("\n" + "="*60)
        print("STRATEGY COMPARISON TABLE")
        print("="*60)
        
        comparison_data = []
        
        for model_name, results in self.backtest_results.items():
            metrics = results['metrics']
            
            row = {
                'Strategy': model_name,
                'Total_Return_%': metrics['total_return'],
                'Annual_Return_%': metrics['annualized_return'],
                'Volatility_%': metrics['volatility'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown_%': metrics['max_drawdown'],
                'Num_Trades': metrics['num_trades'],
                'Win_Rate_%': metrics['win_rate'],
                'Profit_Factor': metrics['profit_factor']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values('Sharpe_Ratio', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        save_path = f"{self.outputs_dir}/backtest_results/strategy_comparison.csv"
        comparison_df.to_csv(save_path, index=False)
        print(f"\nSaved comparison table: {save_path}")
        
        return comparison_df
    
    def save_results(self):
        """Save all backtest results to files."""
        print("\nSaving backtest results...")
        
        for model_name, results in self.backtest_results.items():
            # Save equity curve
            equity_path = f"{self.outputs_dir}/backtest_results/{model_name}_equity.csv"
            results['equity_curve'].to_csv(equity_path, index=False)
            
            # Save trades if available
            if len(results['trades']) > 0:
                trades_path = f"{self.outputs_dir}/backtest_results/{model_name}_trades.csv"
                results['trades'].to_csv(trades_path, index=False)
        
        # Save metrics as JSON
        metrics_dict = {
            name: results['metrics'] 
            for name, results in self.backtest_results.items()
        }
        
        metrics_path = f"{self.outputs_dir}/backtest_results/performance_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        print("All results saved")


def main():
    """
    Main execution function for backtesting.
    """
    print("="*60)
    print("BACKTESTING PIPELINE")
    print("="*60)
    
    # Initialize backtester
    backtester = Backtester(
        signals_path='outputs/predictions/signals.csv',
        data_dir='data/processed',
        outputs_dir='outputs',
        initial_capital=10000,
        position_size=0.1  # 10% of capital per position
    )
    
    # Load signals
    backtester.load_signals()
    
    # Run all backtests
    backtester.run_all_backtests()
    
    # Generate visualizations
    backtester.plot_equity_curves()
    backtester.plot_drawdowns()
    backtester.plot_returns_distribution()
    
    # Create comparison table
    comparison_df = backtester.create_comparison_table()
    
    # Save results
    backtester.save_results()
    
    print("\n" + "="*60)
    print("BACKTESTING COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - outputs/backtest_results/equity_curves.png")
    print("  - outputs/backtest_results/strategy_comparison.csv")
    print("  - outputs/backtest_results/performance_metrics.json")
    print("\nNext step:")
    print("  - Review comparison.md for final analysis")
    
    return backtester


if __name__ == "__main__":
    backtester = main()