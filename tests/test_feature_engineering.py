import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path if using nested structure
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_engineering import FeatureEngineer


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for FeatureEngineer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used across all tests."""
        cls.fe = FeatureEngineer(
            data_path='data/raw/market_data_ml.csv',
            config_path='data/raw/features_config.json'
        )
        cls.fe.load_data()
        cls.fe.load_config()
    
    def test_data_loading(self):
        """Test that data loads correctly."""
        self.assertIsNotNone(self.fe.df)
        self.assertGreater(len(self.fe.df), 0)
        self.assertIn('close', self.fe.df.columns)
        self.assertIn('ticker', self.fe.df.columns)
        print("Data loading test passed")
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        self.assertIsNotNone(self.fe.features_config)
        self.assertIn('features', self.fe.features_config)
        self.assertIn('label', self.fe.features_config)
        self.assertEqual(self.fe.features_config['label'], 'direction')
        print("Config loading test passed")
    
    def test_return_calculation(self):
        """Test that return features are calculated correctly."""
        df = self.fe.calculate_returns(self.fe.df.copy())
        
        # Check that return columns exist
        self.assertIn('return_1d', df.columns)
        self.assertIn('return_3d', df.columns)
        self.assertIn('return_5d', df.columns)
        
        # Check that returns are in reasonable range (-1 to 1 for most cases)
        self.assertTrue((df['return_1d'].dropna().abs() < 1).all() or 
                       (df['return_1d'].dropna().abs() < 1).sum() / len(df['return_1d'].dropna()) > 0.95)
        print("Return calculation test passed")
    
    def test_sma_calculation(self):
        """Test that SMA features are calculated correctly."""
        df = self.fe.calculate_sma(self.fe.df.copy(), windows=[5, 10])
        
        # Check that SMA columns exist
        self.assertIn('sma_5', df.columns)
        self.assertIn('sma_10', df.columns)
        
        # Check that SMAs are positive (for positive prices)
        self.assertTrue((df['sma_5'].dropna() > 0).all())
        self.assertTrue((df['sma_10'].dropna() > 0).all())
        
        # Check that SMA_10 has more NaN at the beginning than SMA_5
        self.assertGreater(df['sma_10'].isna().sum(), df['sma_5'].isna().sum())
        print("SMA calculation test passed")
    
    def test_rsi_calculation(self):
        """Test that RSI is calculated correctly."""
        df = self.fe.calculate_rsi(self.fe.df.copy(), period=14)
        
        # Check that RSI column exists
        self.assertIn('rsi_14', df.columns)
        
        # Check that RSI is in valid range (0-100)
        rsi_values = df['rsi_14'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())
        print("RSI calculation test passed")
    
    def test_macd_calculation(self):
        """Test that MACD is calculated correctly."""
        df = self.fe.calculate_macd(self.fe.df.copy())
        
        # Check that MACD columns exist
        self.assertIn('macd', df.columns)
        self.assertIn('macd_signal', df.columns)
        self.assertIn('macd_histogram', df.columns)
        
        # Check that histogram = macd - signal
        macd_diff = df['macd'] - df['macd_signal']
        self.assertTrue(np.allclose(macd_diff.dropna(), 
                                   df['macd_histogram'].dropna(), 
                                   rtol=1e-5))
        print("MACD calculation test passed")
    
    def test_label_creation(self):
        """Test that labels are created correctly."""
        df = self.fe.calculate_returns(self.fe.df.copy())
        df = self.fe.create_labels(df)
        
        # Check that label column exists
        self.assertIn('direction', df.columns)
        
        # Check that labels are binary (0 or 1)
        self.assertTrue(df['direction'].dropna().isin([0, 1]).all())
        
        # Check that we have both classes
        unique_labels = df['direction'].dropna().unique()
        self.assertEqual(len(unique_labels), 2)
        print("Label creation test passed")
    
    def test_feature_engineering_pipeline(self):
        """Test the complete feature engineering pipeline."""
        fe_test = FeatureEngineer(
            data_path='data/raw/market_data_ml.csv',
            config_path='data/raw/features_config.json'
        )
        
        # Run full pipeline
        df_features = fe_test.engineer_features()
        
        # Check that all required features exist
        for feature in fe_test.features_config['features']:
            self.assertIn(feature, df_features.columns)
        
        # Check that label exists
        self.assertIn(fe_test.features_config['label'], df_features.columns)
        
        # Check that there are no NaN in required columns
        required_cols = fe_test.features_config['features'] + [fe_test.features_config['label']]
        self.assertEqual(df_features[required_cols].isna().sum().sum(), 0)
        print("Full pipeline test passed")
    
    def test_data_split(self):
        """Test train/test split functionality."""
        fe_test = FeatureEngineer(
            data_path='data/raw/market_data_ml.csv',
            config_path='data/raw/features_config.json'
        )
        fe_test.engineer_features()
        
        # Split data
        X_train, X_test, y_train, y_test, train_df, test_df = fe_test.split_data(test_size=0.2)
        
        # Check shapes
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check that train is larger than test
        self.assertGreater(len(X_train), len(X_test))
        
        # Check that train comes before test (time-based split)
        self.assertLess(train_df['date'].max(), test_df['date'].min() or 
                       pd.Timestamp('2100-01-01'))  # Handle edge cases
        
        # Check that features are scaled (mean ≈ 0, std ≈ 1)
        self.assertTrue(np.abs(X_train.mean().mean()) < 0.1)
        self.assertTrue(np.abs(X_train.std().mean() - 1.0) < 0.1)
        print("Data split test passed")
    
    def test_feature_shapes(self):
        """Test that feature arrays have correct shapes."""
        fe_test = FeatureEngineer(
            data_path='data/raw/market_data_ml.csv',
            config_path='data/raw/features_config.json'
        )
        fe_test.engineer_features()
        X_train, X_test, y_train, y_test, _, _ = fe_test.split_data()
        
        # Check number of features matches config
        expected_features = len(fe_test.features_config['features'])
        self.assertEqual(X_train.shape[1], expected_features)
        self.assertEqual(X_test.shape[1], expected_features)
        print("Feature shape test passed")


def run_tests():
    """Run all tests and display results."""
    print("="*60)
    print("RUNNING FEATURE ENGINEERING TESTS")
    print("="*60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFeatureEngineering)
    
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