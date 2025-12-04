import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path if using nested structure
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from backtest import Backtester


class TestBacktest(unittest.TestCase):
    """Test suite for Backtester class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used across all tests."""
        # Make sure we have signals
        if not os.path.exists('outputs/predictions/signals.csv'):
            print("\nWarning: No signals found. Running signal_generator.py first...")
            from signal_generator import SignalGenerator
            sg = SignalGenerator()
            sg.load_models()
            X, y, df = sg.load_data('test')
            predictions = sg.generate_predictions(X)
            sg.create_signals(predictions, df)
            sg.save_signals('signals.csv')
        
        cls.backtester = Backtester(
            signals_path='outputs/predictions/signals.csv',
            data_dir='data/processed',
            outputs_dir='outputs',
            initial_capital=10000,
            position_size=0.1
        )
    
    def test_signal_loading(self):
        """Test that signals load correctly."""
        signals_df = self.backtester.load_signals()
        
        self.assertIsNotNone(signals_df)
        self.assertGreater(len(signals_df), 0)
        
        # Check expected columns
        self.assertIn('date', signals_df.columns)
        self.assertIn('ticker', signals_df.columns)
        
        # Check that at least one signal column exists
        signal_cols = [col for col in signals_df.columns if col.endswith('_signal')]
        self.assertGreater(len(signal_cols), 0)
        
        print("Signal loading test passed")
    
    def test_price_data_loading(self):
        """Test that price data loads correctly."""
        price_df = self.backtester.load_price_data()
        
        self.assertIsNotNone(price_df)
        self.assertGreater(len(price_df), 0)
        
        # Check expected columns
        self.assertIn('date', price_df.columns)
        self.assertIn('ticker', price_df.columns)
        self.assertIn('close', price_df.columns)
        
        # Check that prices are positive
        self.assertTrue((price_df['close'] > 0).all())
        
        print("Price data loading test passed")
    
    def test_backtest_execution(self):
        """Test that backtest runs without errors."""
        self.backtester.load_signals()
        
        # Find a signal column
        signal_cols = [col for col in self.backtester.signals_df.columns 
                      if col.endswith('_signal')]
        
        if len(signal_cols) > 0:
            signal_col = signal_cols[0]
            model_name = signal_col.replace('_signal', '')
            
            results = self.backtester.run_backtest(signal_col, model_name)
            
            # Check results structure
            self.assertIn('model_name', results)
            self.assertIn('equity_curve', results)
            self.assertIn('trades', results)
            self.assertIn('metrics', results)
            
            # Check equity curve
            self.assertGreater(len(results['equity_curve']), 0)
            self.assertIn('date', results['equity_curve'].columns)
            self.assertIn('portfolio_value', results['equity_curve'].columns)
            
            # Check that portfolio values are positive
            self.assertTrue((results['equity_curve']['portfolio_value'] > 0).all())
            
            print("Backtest execution test passed")
        else:
            print("No signal columns found, skipping test")
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        # Create sample equity curve
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        portfolio_values = np.linspace(10000, 12000, 100)  # Steady growth
        
        equity_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        # Create empty trades dataframe
        trades_df = pd.DataFrame()
        
        metrics = self.backtester.calculate_metrics(equity_df, trades_df)
        
        # Check that all expected metrics exist
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'num_trades'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that total return is positive (since we have growth)
        self.assertGreater(metrics['total_return'], 0)
        
        # Check that metrics are reasonable values
        self.assertGreater(metrics['total_return'], -100)  # Not less than -100%
        self.assertLess(metrics['total_return'], 10000)   # Not more than 10000%
        
        print("Metrics calculation test passed")
    
    def test_buy_and_hold_benchmark(self):
        """Test that buy-and-hold benchmark runs correctly."""
        self.backtester.load_signals()
        results = self.backtester.run_buy_and_hold_benchmark()
        
        self.assertIsNotNone(results)
        self.assertEqual(results['model_name'], 'Buy-and-Hold')
        
        # Check equity curve
        self.assertGreater(len(results['equity_curve']), 0)
        
        # Check metrics
        self.assertIn('total_return', results['metrics'])
        self.assertIn('sharpe_ratio', results['metrics'])
        
        print("Buy-and-hold benchmark test passed")
    
    def test_equity_curve_continuity(self):
        """Test that equity curve is continuous (no gaps)."""
        self.backtester.load_signals()
        
        signal_cols = [col for col in self.backtester.signals_df.columns 
                      if col.endswith('_signal')]
        
        if len(signal_cols) > 0:
            results = self.backtester.run_backtest(signal_cols[0], 'test_model')
            
            equity_df = results['equity_curve']
            
            # Check that dates are sorted
            self.assertTrue(equity_df['date'].is_monotonic_increasing)
            
            # Check that there are no duplicate dates
            self.assertEqual(len(equity_df), equity_df['date'].nunique())
            
            print("Equity curve continuity test passed")
        else:
            print("No signal columns found, skipping test")
    
    def test_initial_capital(self):
        """Test that initial portfolio value equals initial capital."""
        self.backtester.load_signals()
        
        signal_cols = [col for col in self.backtester.signals_df.columns 
                      if col.endswith('_signal')]
        
        if len(signal_cols) > 0:
            results = self.backtester.run_backtest(signal_cols[0], 'test_model')
            
            # First portfolio value should be close to initial capital
            # (may differ slightly due to position opening)
            first_value = results['equity_curve']['portfolio_value'].iloc[0]
            self.assertAlmostEqual(first_value, self.backtester.initial_capital, 
                                 delta=self.backtester.initial_capital * 0.2)
            
            print("Initial capital test passed")
        else:
            print("No signal columns found, skipping test")
    
    def test_position_sizing(self):
        """Test that position sizing is respected."""
        self.backtester.load_signals()
        
        signal_cols = [col for col in self.backtester.signals_df.columns 
                      if col.endswith('_signal')]
        
        if len(signal_cols) > 0:
            results = self.backtester.run_backtest(signal_cols[0], 'test_model')
            
            trades_df = results['trades']
            
            if len(trades_df) > 0:
                # Check that buy trades don't exceed position size limit
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                
                if len(buy_trades) > 0:
                    # Position size should be around position_size * capital
                    expected_size = self.backtester.initial_capital * self.backtester.position_size
                    
                    # Check first trade (when we have full capital)
                    first_trade_value = buy_trades.iloc[0]['value']
                    
                    # Should be close to expected size (within 20% tolerance)
                    self.assertLess(first_trade_value, expected_size * 1.2)
                    
                    print("Position sizing test passed")
            else:
                print("No trades executed, skipping test")
        else:
            print("No signal columns found, skipping test")
    
    def test_comparison_table(self):
        """Test that comparison table is created correctly."""
        self.backtester.load_signals()
        
        # Run at least one backtest
        signal_cols = [col for col in self.backtester.signals_df.columns 
                      if col.endswith('_signal')]
        
        if len(signal_cols) > 0:
            self.backtester.run_backtest(signal_cols[0], 'test_model')
            
            comparison_df = self.backtester.create_comparison_table()
            
            # Check that dataframe exists and has data
            self.assertIsNotNone(comparison_df)
            self.assertGreater(len(comparison_df), 0)
            
            # Check expected columns
            expected_cols = ['Strategy', 'Total_Return_%', 'Sharpe_Ratio', 'Max_Drawdown_%']
            for col in expected_cols:
                self.assertIn(col, comparison_df.columns)
            
            print("Comparison table test passed")
        else:
            print("No signal columns found, skipping test")
    
    def test_sharpe_ratio_validity(self):
        """Test that Sharpe ratio is calculated correctly."""
        # Create sample data with known properties
        dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 1 year
        
        # Portfolio with 10% annual return and 15% volatility
        np.random.seed(42)
        daily_returns = np.random.normal(0.1/252, 0.15/np.sqrt(252), 252)
        portfolio_values = 10000 * np.cumprod(1 + daily_returns)
        
        equity_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        trades_df = pd.DataFrame()
        
        metrics = self.backtester.calculate_metrics(equity_df, trades_df)
        
        # Sharpe ratio should be roughly annualized_return / volatility
        # For our data: ~10% / 15% = 0.67
        expected_sharpe = metrics['annualized_return'] / metrics['volatility']
        
        self.assertAlmostEqual(metrics['sharpe_ratio'], expected_sharpe, places=5)
        
        print("Sharpe ratio validity test passed")


def run_tests():
    """Run all tests and display results."""
    print("="*60)
    print("RUNNING BACKTEST TESTS")
    print("="*60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBacktest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED")
    
    return result


if __name__ == "__main__":
    run_tests()