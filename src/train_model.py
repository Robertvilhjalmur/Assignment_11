import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, cross_validate
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, install if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    """
    A class to handle training and evaluation of multiple ML models.
    """
    
    def __init__(self, params_path='data/raw/model_params.json',
                 data_dir='data/processed',
                 models_dir='models',
                 outputs_dir='outputs'):
        """
        Initialize the ModelTrainer.
        
        Parameters:
        -----------
        params_path : str
            Path to model parameters JSON file
        data_dir : str
            Directory containing processed data
        models_dir : str
            Directory to save trained models
        outputs_dir : str
            Directory to save outputs and visualizations
        """
        self.params_path = params_path
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        self.params = None
        self.models = {}
        self.results = {}
        
        # Create output directories
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{outputs_dir}/visualizations").mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load processed training and test data."""
        print("Loading processed data...")
        
        train_df = pd.read_csv(f"{self.data_dir}/train_data.csv")
        test_df = pd.read_csv(f"{self.data_dir}/test_data.csv")
        
        # Load feature info
        with open(f"{self.data_dir}/feature_info.json", 'r') as f:
            feature_info = json.load(f)
        
        self.feature_cols = feature_info['feature_columns']
        self.label_col = feature_info['label_column']
        
        # Extract features and labels
        self.X_train = train_df[self.feature_cols]
        self.y_train = train_df[self.label_col]
        self.X_test = test_df[self.feature_cols]
        self.y_test = test_df[self.label_col]
        
        # Store full dataframes for later analysis
        self.train_df = train_df
        self.test_df = test_df
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Class distribution (train): {self.y_train.value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def load_params(self):
        """Load model hyperparameters from JSON."""
        print(f"\nLoading model parameters from {self.params_path}...")
        
        with open(self.params_path, 'r') as f:
            self.params = json.load(f)
        
        print(f"Loaded parameters for: {list(self.params.keys())}")
        return self.params
    
    def initialize_models(self):
        """Initialize models with specified hyperparameters."""
        print("\nInitializing models...")
        
        # Logistic Regression
        if 'LogisticRegression' in self.params:
            self.models['LogisticRegression'] = LogisticRegression(
                **self.params['LogisticRegression'],
                random_state=42
            )
            print("Logistic Regression initialized")
        
        # Random Forest
        if 'RandomForestClassifier' in self.params:
            self.models['RandomForestClassifier'] = RandomForestClassifier(
                **self.params['RandomForestClassifier'],
                random_state=42,
                n_jobs=-1
            )
            print("Random Forest initialized")
        
        # XGBoost
        if 'XGBClassifier' in self.params and XGBOOST_AVAILABLE:
            self.models['XGBClassifier'] = XGBClassifier(
                **self.params['XGBClassifier'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            print("XGBoost initialized")
        elif 'XGBClassifier' in self.params and not XGBOOST_AVAILABLE:
            print("⚠ XGBoost skipped (not installed)")
        
        return self.models
    
    def train_model(self, model_name, model):
        """
        Train a single model and evaluate it.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn estimator
            Model instance to train
            
        Returns:
        --------
        dict
            Dictionary containing model and evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        # Train model
        model.fit(self.X_train, self.y_train)
        print(f"{model_name} training complete")
        
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Get prediction probabilities
        y_train_proba = model.predict_proba(self.X_train)[:, 1]
        y_test_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(self.y_train, y_train_pred, y_train_proba)
        test_metrics = self.calculate_metrics(self.y_test, y_test_pred, y_test_proba)
        
        # Perform cross-validation
        cv_scores = self.cross_validate_model(model, model_name)
        
        # Store results
        results = {
            'model': model,
            'model_name': model_name,
            'train_predictions': y_train_pred,
            'test_predictions': y_test_pred,
            'train_probabilities': y_train_proba,
            'test_probabilities': y_test_proba,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores
        }
        
        # Print results
        self.print_results(model_name, train_metrics, test_metrics, cv_scores)
        
        return results
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate classification metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like
            Prediction probabilities
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        return metrics
    
    def cross_validate_model(self, model, model_name):
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to cross-validate
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        print(f"Performing 5-fold cross-validation for {model_name}...")
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(
            model, self.X_train, self.y_train,
            cv=5,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        
        cv_scores = {
            'accuracy': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'precision': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'roc_auc': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std()
        }
        
        return cv_scores
    
    def print_results(self, model_name, train_metrics, test_metrics, cv_scores):
        """Print formatted results for a model."""
        print(f"\n{model_name} Results:")
        print("-" * 60)
        
        print("\nTraining Set:")
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize():12s}: {value:.4f}")
        
        print("\nTest Set:")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize():12s}: {value:.4f}")
        
        print("\nCross-Validation (5-fold):")
        print(f"  Accuracy:  {cv_scores['accuracy']:.4f} ± {cv_scores['accuracy_std']:.4f}")
        print(f"  Precision: {cv_scores['precision']:.4f} ± {cv_scores['precision_std']:.4f}")
        print(f"  Recall:    {cv_scores['recall']:.4f} ± {cv_scores['recall_std']:.4f}")
        print(f"  F1 Score:  {cv_scores['f1']:.4f} ± {cv_scores['f1_std']:.4f}")
        print(f"  ROC AUC:   {cv_scores['roc_auc']:.4f} ± {cv_scores['roc_auc_std']:.4f}")
    
    def train_all_models(self):
        """Train all initialized models."""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            results = self.train_model(model_name, model)
            self.results[model_name] = results
        
        return self.results
    
    def plot_confusion_matrix(self, model_name, y_true, y_pred):
        """
        Plot confusion matrix for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Down', 'Up'],
                    yticklabels=['Down', 'Up'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add metrics to plot
        accuracy = accuracy_score(y_true, y_pred)
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        save_path = f"{self.outputs_dir}/visualizations/confusion_matrix_{model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, model_name, model):
        """
        Plot feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn estimator
            Trained model
        """
        # Only for tree-based models
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠ {model_name} does not have feature_importances_")
            return
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        save_path = f"{self.outputs_dir}/visualizations/feature_importance_{model_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance: {save_path}")
        plt.close()
        
        return feature_importance_df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models on the same plot."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['test_probabilities'])
            auc_score = results['test_metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = f"{self.outputs_dir}/visualizations/roc_curves_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves: {save_path}")
        plt.close()
    
    def create_comparison_table(self):
        """Create a comparison table of all models."""
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Train_Accuracy': results['train_metrics']['accuracy'],
                'Test_Accuracy': results['test_metrics']['accuracy'],
                'Test_Precision': results['test_metrics']['precision'],
                'Test_Recall': results['test_metrics']['recall'],
                'Test_F1': results['test_metrics']['f1'],
                'Test_ROC_AUC': results['test_metrics']['roc_auc'],
                'CV_Accuracy': results['cv_scores']['accuracy'],
                'CV_Accuracy_Std': results['cv_scores']['accuracy_std']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by test accuracy
        comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON TABLE")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        save_path = f"{self.outputs_dir}/model_comparison.csv"
        comparison_df.to_csv(save_path, index=False)
        print(f"\nSaved comparison table: {save_path}")
        
        return comparison_df
    
    def save_models(self):
        """Save all trained models to disk."""
        print("\nSaving trained models...")
        
        for model_name, results in self.results.items():
            model = results['model']
            save_path = f"{self.models_dir}/{model_name}.pkl"
            
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Saved {model_name} to {save_path}")
        
        # Save model metadata
        metadata = {}
        for model_name, results in self.results.items():
            metadata[model_name] = {
                'test_accuracy': results['test_metrics']['accuracy'],
                'test_f1': results['test_metrics']['f1'],
                'test_roc_auc': results['test_metrics']['roc_auc'],
                'cv_accuracy': results['cv_scores']['accuracy']
            }
        
        metadata_path = f"{self.models_dir}/model_metrics.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved model metadata to {metadata_path}")
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        for model_name, results in self.results.items():
            # Confusion matrix
            self.plot_confusion_matrix(
                model_name,
                self.y_test,
                results['test_predictions']
            )
            
            # Feature importance (for applicable models)
            self.plot_feature_importance(model_name, results['model'])
        
        # ROC curves
        self.plot_roc_curves()
        
        print("\nAll visualizations generated")


def main():
    """
    Main execution function for model training.
    """
    print("="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer(
        params_path='data/raw/model_params.json',
        data_dir='data/processed',
        models_dir='models',
        outputs_dir='outputs'
    )
    
    # Load data and parameters
    trainer.load_data()
    trainer.load_params()
    
    # Initialize and train models
    trainer.initialize_models()
    trainer.train_all_models()
    
    # Generate visualizations
    trainer.generate_visualizations()
    
    # Create comparison table
    comparison_df = trainer.create_comparison_table()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review model performance in outputs/model_comparison.csv")
    print("2. Check visualizations in outputs/visualizations/")
    print("3. Run signal_generator.py to generate trading signals")
    
    return trainer


if __name__ == "__main__":
    trainer = main()