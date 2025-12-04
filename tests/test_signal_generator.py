import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from signal_generator import SignalGenerator


class TestSignalGenerator(unittest.TestCase):
    """Test suite for SignalGenerator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used across all tests."""
        # Make sure we have trained models
        if not os.path.exists('models'):
            print("\nWarning: No trained models found. Running train_model.py first...")
            from train_model import ModelTrainer
            trainer = ModelTrainer()
            trainer.load_data()
            trainer.load_params()
            trainer.initialize_models()
            trainer.train_all_models()
            trainer.save_models()
        
        cls.sg = SignalGenerator(
            models_dir='models',
            data_dir='data/processed',
            outputs_dir='outputs'
        )
    
    def test_model_loading(self):
        """Test that models load correctly."""
        models = self.sg.load_models()
        
        self.assertIsNotNone(models)
        self.assertGreater(len(models), 0)
        
        # Check that at least one model was loaded
        self.assertIn('LogisticRegression', models)
        
        print("Model loading test passed")
    
    def test_data_loading(self):
        """Test that data loads correctly."""
        X, y, df = self.sg.load_data(data_type='test')
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(df)
        self.assertGreater(len(X), 0)
        
        # Check that dataframe has expected columns
        self.assertIn('date', df.columns)
        self.assertIn('ticker', df.columns)
        
        print("Data loading test passed")
    
    def test_prediction_generation(self):
        """Test that predictions are generated correctly."""
        self.sg.load_models()
        X, y, df = self.sg.load_data(data_type='test')
        
        predictions = self.sg.generate_predictions(X)
        
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        
        # Check that predictions have correct structure
        for model_name, preds in predictions.items():
            self.assertIn('labels', preds)
            self.assertIn('probabilities', preds)
            
            # Check shapes
            self.assertEqual(len(preds['labels']), len(X))
            self.assertEqual(len(preds['probabilities']), len(X))
            
            # Check that labels are binary
            unique_labels = np.unique(preds['labels'])
            self.assertTrue(all(label in [0, 1] for label in unique_labels))
            
            # Check that probabilities are in [0, 1]
            self.assertTrue(all(0 <= p <= 1 for p in preds['probabilities']))
        
        print("Prediction generation test passed")
    
    def test_threshold_signal(self):
        """Test threshold signal generation."""
        probabilities = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        
        signals = self.sg._threshold_signal(probabilities, threshold=0.5)
        
        # Check that signals are in {-1, 0, 1}
        unique_signals = np.unique(signals)
        self.assertTrue(all(s in [-1, 0, 1] for s in unique_signals))
        
        # Check logic
        self.assertEqual(signals[0], -1)  # 0.2 < 0.5 → SELL
        self.assertEqual(signals[2], 0)   # 0.5 = 0.5 → HOLD
        self.assertEqual(signals[4], 1)   # 0.8 > 0.5 → BUY
        
        print("Threshold signal test passed")
    
    def test_signal_creation(self):
        """Test that signals are created correctly."""
        self.sg.load_models()
        X, y, df = self.sg.load_data(data_type='test')
        predictions = self.sg.generate_predictions(X)
        
        signals_df = self.sg.create_signals(predictions, df, strategy='threshold', threshold=0.5)
        
        self.assertIsNotNone(signals_df)
        self.assertEqual(len(signals_df), len(df))
        
        # Check that signal columns exist
        for model_name in predictions.keys():
            self.assertIn(f'{model_name}_prediction', signals_df.columns)
            self.assertIn(f'{model_name}_probability', signals_df.columns)
            self.assertIn(f'{model_name}_signal', signals_df.columns)
            self.assertIn(f'{model_name}_action', signals_df.columns)
        
        # Check that ensemble columns exist if multiple models
        if len(predictions) > 1:
            self.assertIn('ensemble_signal', signals_df.columns)
            self.assertIn('ensemble_action', signals_df.columns)
            self.assertIn('ensemble_confidence', signals_df.columns)
        
        # Check signal values
        signal_col = f'{list(predictions.keys())[0]}_signal'
        unique_signals = signals_df[signal_col].unique()
        self.assertTrue(all(s in [-1, 0, 1] for s in unique_signals))
        
        print("Signal creation test passed")
    
    def test_signal_analysis(self):
        """Test signal analysis functionality."""
        self.sg.load_models()
        X, y, df = self.sg.load_data(data_type='test')
        predictions = self.sg.generate_predictions(X)
        self.sg.create_signals(predictions, df)
        
        stats = self.sg.analyze_signals()
        
        self.assertIsNotNone(stats)
        self.assertGreater(len(stats), 0)
        
        # Check that stats have expected structure
        for model_name, model_stats in stats.items():
            self.assertIn('buy_signals', model_stats)
            self.assertIn('hold_signals', model_stats)
            self.assertIn('sell_signals', model_stats)
            
            # Check that counts sum to total
            total = len(self.sg.signals_df)
            sum_signals = (model_stats['buy_signals'] + 
                          model_stats['hold_signals'] + 
                          model_stats['sell_signals'])
            self.assertEqual(sum_signals, total)
        
        print("Signal analysis test passed")
    
    def test_signal_filtering(self):
        """Test filtering signals by action."""
        self.sg.load_models()
        X, y, df = self.sg.load_data(data_type='test')
        predictions = self.sg.generate_predictions(X)
        self.sg.create_signals(predictions, df)
        
        # Test filtering BUY signals
        buy_signals = self.sg.filter_signals_by_action('BUY')
        
        if buy_signals is not None and len(buy_signals) > 0:
            # Check that all filtered signals are BUY
            action_col = 'ensemble_action' if 'ensemble_action' in buy_signals.columns else None
            if action_col:
                self.assertTrue(all(buy_signals[action_col] == 'BUY'))
        
        print("Signal filtering test passed")
    
    def test_latest_signals(self):
        """Test retrieving latest signals."""
        self.sg.load_models()
        X, y, df = self.sg.load_data(data_type='test')
        predictions = self.sg.generate_predictions(X)
        self.sg.create_signals(predictions, df)
        
        latest = self.sg.get_latest_signals(n=5)
        
        self.assertIsNotNone(latest)
        self.assertLessEqual(len(latest), 5)
        
        # Check that dates are sorted in descending order
        if len(latest) > 1:
            dates = pd.to_datetime(latest['date'])
            self.assertTrue((dates.diff().dropna() <= pd.Timedelta(0)).all())
        
        print("Latest signals test passed")
    
    def test_signal_saving(self):
        """Test that signals can be saved."""
        self.sg.load_models()
        X, y, df = self.sg.load_data(data_type='test')
        predictions = self.sg.generate_predictions(X)
        self.sg.create_signals(predictions, df)
        
        save_path = self.sg.save_signals('test_signals.csv')
        
        # Check that file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Check that file can be read back
        loaded_signals = pd.read_csv(save_path)
        self.assertEqual(len(loaded_signals), len(self.sg.signals_df))
        
        print("Signal saving test passed")
    
    def test_ensemble_confidence(self):
        """Test that ensemble confidence is calculated correctly."""
        self.sg.load_models()
        
        # Skip test if only one model
        if len(self.sg.models) < 2:
            print("⚠ Skipping ensemble test (need at least 2 models)")
            return
        
        X, y, df = self.sg.load_data(data_type='test')
        predictions = self.sg.generate_predictions(X)
        signals_df = self.sg.create_signals(predictions, df)
        
        # Check that confidence is in [0, 1]
        if 'ensemble_confidence' in signals_df.columns:
            confidence = signals_df['ensemble_confidence']
            self.assertTrue(all(0 <= c <= 1 for c in confidence))
            
            # Check that perfect agreement = 1.0 confidence
            # (when all models agree, confidence should be 1.0)
            signal_cols = [f'{name}_signal' for name in predictions.keys()]
            perfect_agreement = signals_df[signal_cols].apply(
                lambda row: len(set(row)) == 1, axis=1
            )
            if perfect_agreement.any():
                self.assertTrue(
                    all(signals_df.loc[perfect_agreement, 'ensemble_confidence'] == 1.0)
                )
        
        print("Ensemble confidence test passed")


def run_tests():
    """Run all tests and display results."""
    print("="*60)
    print("RUNNING SIGNAL GENERATOR TESTS")
    print("="*60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSignalGenerator)
    
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