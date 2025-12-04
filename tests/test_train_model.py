import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent / 'src'))

from train_model import ModelTrainer


class TestModelTraining(unittest.TestCase):
    """Test suite for ModelTrainer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used across all tests."""
        # First, make sure processed data exists
        if not os.path.exists('data/processed/train_data.csv'):
            print("\n⚠ Warning: Processed data not found. Running feature engineering first...")
            from feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            fe.engineer_features()
            fe.split_data()
            _, _, _, _, train_df, test_df = fe.split_data()
            fe.save_processed_data(train_df, test_df)
        
        cls.trainer = ModelTrainer(
            params_path='data/raw/model_params.json',
            data_dir='data/processed',
            models_dir='models',
            outputs_dir='outputs'
        )
    
    def test_data_loading(self):
        """Test that data loads correctly."""
        X_train, X_test, y_train, y_test = self.trainer.load_data()
        
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        
        # Check shapes match
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check that train is larger than test
        self.assertGreater(len(X_train), len(X_test))
        
        print("Data loading test passed")
    
    def test_params_loading(self):
        """Test that model parameters load correctly."""
        params = self.trainer.load_params()
        
        self.assertIsNotNone(params)
        self.assertIsInstance(params, dict)
        
        # Check expected model types
        expected_models = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier']
        for model_name in expected_models:
            self.assertIn(model_name, params)
        
        print("Parameters loading test passed")
    
    def test_model_initialization(self):
        """Test that models initialize correctly."""
        self.trainer.load_data()
        self.trainer.load_params()
        models = self.trainer.initialize_models()
        
        self.assertIsNotNone(models)
        self.assertGreater(len(models), 0)
        
        # Check that at least Logistic Regression and Random Forest are initialized
        self.assertIn('LogisticRegression', models)
        self.assertIn('RandomForestClassifier', models)
        
        print("Model initialization test passed")
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        # Create dummy predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        y_proba = np.array([0.2, 0.8, 0.9, 0.3, 0.4, 0.6, 0.7, 0.85, 0.1, 0.25])
        
        metrics = self.trainer.calculate_metrics(y_true, y_pred, y_proba)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)
        
        print("Metrics calculation test passed")
    
    def test_single_model_training(self):
        """Test training a single model."""
        self.trainer.load_data()
        self.trainer.load_params()
        self.trainer.initialize_models()
        
        # Train just Logistic Regression
        model_name = 'LogisticRegression'
        model = self.trainer.models[model_name]
        
        results = self.trainer.train_model(model_name, model)
        
        # Check that results contain expected keys
        expected_keys = [
            'model', 'model_name', 'train_predictions', 'test_predictions',
            'train_probabilities', 'test_probabilities', 
            'train_metrics', 'test_metrics', 'cv_scores'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check that predictions have correct shape
        self.assertEqual(len(results['test_predictions']), len(self.trainer.y_test))
        
        # Check that metrics are reasonable
        self.assertGreater(results['test_metrics']['accuracy'], 0.4)
        self.assertLess(results['test_metrics']['accuracy'], 1.0)
        
        print("Single model training test passed")
    
    def test_model_predictions_shape(self):
        """Test that model predictions have correct shape."""
        self.trainer.load_data()
        self.trainer.load_params()
        self.trainer.initialize_models()
        
        # Train one model
        model_name = 'LogisticRegression'
        model = self.trainer.models[model_name]
        results = self.trainer.train_model(model_name, model)
        
        # Check prediction shapes
        self.assertEqual(
            len(results['train_predictions']),
            len(self.trainer.y_train)
        )
        self.assertEqual(
            len(results['test_predictions']),
            len(self.trainer.y_test)
        )
        self.assertEqual(
            len(results['train_probabilities']),
            len(self.trainer.y_train)
        )
        self.assertEqual(
            len(results['test_probabilities']),
            len(self.trainer.y_test)
        )
        
        # Check that predictions are binary
        unique_preds = np.unique(results['test_predictions'])
        self.assertTrue(all(p in [0, 1] for p in unique_preds))
        
        # Check that probabilities are in [0, 1]
        self.assertTrue(all(0 <= p <= 1 for p in results['test_probabilities']))
        
        print("Model predictions shape test passed")
    
    def test_cross_validation(self):
        """Test that cross-validation works correctly."""
        self.trainer.load_data()
        self.trainer.load_params()
        self.trainer.initialize_models()
        
        model_name = 'LogisticRegression'
        model = self.trainer.models[model_name]
        
        cv_scores = self.trainer.cross_validate_model(model, model_name)
        
        # Check that CV scores contain expected metrics
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, cv_scores)
            self.assertIn(f'{metric}_std', cv_scores)
        
        # Check that scores are reasonable
        self.assertGreater(cv_scores['accuracy'], 0.3)
        self.assertLess(cv_scores['accuracy'], 1.0)
        
        print("Cross-validation test passed")
    
    def test_comparison_table_creation(self):
        """Test that comparison table is created correctly."""
        self.trainer.load_data()
        self.trainer.load_params()
        self.trainer.initialize_models()
        
        # Train at least two models
        for model_name in ['LogisticRegression', 'RandomForestClassifier']:
            if model_name in self.trainer.models:
                model = self.trainer.models[model_name]
                results = self.trainer.train_model(model_name, model)
                self.trainer.results[model_name] = results
        
        # Create comparison table
        comparison_df = self.trainer.create_comparison_table()
        
        # Check that dataframe has expected columns
        expected_cols = ['Model', 'Train_Accuracy', 'Test_Accuracy', 
                        'Test_Precision', 'Test_Recall', 'Test_F1']
        for col in expected_cols:
            self.assertIn(col, comparison_df.columns)
        
        # Check that we have results for trained models
        self.assertEqual(len(comparison_df), len(self.trainer.results))
        
        print("Comparison table creation test passed")
    
    def test_model_saving(self):
        """Test that models can be saved."""
        self.trainer.load_data()
        self.trainer.load_params()
        self.trainer.initialize_models()
        
        # Train one model
        model_name = 'LogisticRegression'
        model = self.trainer.models[model_name]
        results = self.trainer.train_model(model_name, model)
        self.trainer.results[model_name] = results
        
        # Save models
        self.trainer.save_models()
        
        # Check that model file exists
        model_path = f"{self.trainer.models_dir}/{model_name}.pkl"
        self.assertTrue(os.path.exists(model_path))
        
        # Check that metadata file exists
        metadata_path = f"{self.trainer.models_dir}/model_metrics.json"
        self.assertTrue(os.path.exists(metadata_path))
        
        print("Model saving test passed")


def run_tests():
    """Run all tests and display results."""
    print("="*60)
    print("RUNNING MODEL TRAINING TESTS")
    print("="*60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestModelTraining)
    
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
        print("\n✗ SOME TESTS FAILED")
    
    return result


if __name__ == "__main__":
    run_tests()