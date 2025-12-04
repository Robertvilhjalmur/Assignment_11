import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SignalGenerator:
    """
    A class to generate trading signals from model predictions.
    """
    
    def __init__(self, models_dir='models', 
                 data_dir='data/processed',
                 outputs_dir='outputs'):
        """
        Initialize the SignalGenerator.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained models
        data_dir : str
            Directory containing processed data
        outputs_dir : str
            Directory to save outputs
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.outputs_dir = outputs_dir
        self.models = {}
        self.signals_df = None
        
        # Create output directories
        Path(f"{outputs_dir}/predictions").mkdir(parents=True, exist_ok=True)
        
    def load_models(self, model_names=None):
        """
        Load trained models from disk.
        
        Parameters:
        -----------
        model_names : list, optional
            List of model names to load. If None, loads all available models.
            
        Returns:
        --------
        dict
            Dictionary of loaded models
        """
        print("Loading trained models...")
        
        if model_names is None:
            # Load all .pkl files in models directory
            model_files = list(Path(self.models_dir).glob('*.pkl'))
            model_names = [f.stem for f in model_files]
        
        for model_name in model_names:
            model_path = f"{self.models_dir}/{model_name}.pkl"
            
            if Path(model_path).exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded {model_name}")
            else:
                print(f"⚠ Model not found: {model_path}")
        
        print(f"\nLoaded {len(self.models)} models")
        return self.models
    
    def load_data(self, data_type='test'):
        """
        Load data for prediction.
        
        Parameters:
        -----------
        data_type : str
            Type of data to load: 'train' or 'test'
            
        Returns:
        --------
        tuple
            (features, labels, full_dataframe)
        """
        print(f"\nLoading {data_type} data...")
        
        # Load data
        df = pd.read_csv(f"{self.data_dir}/{data_type}_data.csv")
        
        # Load feature info
        with open(f"{self.data_dir}/feature_info.json", 'r') as f:
            feature_info = json.load(f)
        
        feature_cols = feature_info['feature_columns']
        label_col = feature_info['label_column']
        
        X = df[feature_cols]
        y = df[label_col] if label_col in df.columns else None
        
        print(f"Loaded {len(df)} samples with {len(feature_cols)} features")
        
        return X, y, df
    
    def generate_predictions(self, X, model_name=None):
        """
        Generate predictions using specified model(s).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
        model_name : str, optional
            Specific model to use. If None, uses all loaded models.
            
        Returns:
        --------
        dict
            Dictionary mapping model names to predictions and probabilities
        """
        predictions = {}
        
        models_to_use = {model_name: self.models[model_name]} if model_name else self.models
        
        print("\nGenerating predictions...")
        for name, model in models_to_use.items():
            pred_labels = model.predict(X)
            pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (UP)
            
            predictions[name] = {
                'labels': pred_labels,
                'probabilities': pred_proba
            }
            print(f"{name}: {len(pred_labels)} predictions generated")
        
        return predictions
    
    def create_signals(self, predictions, df, strategy='threshold', threshold=0.5):
        """
        Convert predictions to trading signals.
        
        Parameters:
        -----------
        predictions : dict
            Dictionary of model predictions
        df : pd.DataFrame
            Original dataframe with date, ticker, etc.
        strategy : str
            Signal generation strategy:
            - 'threshold': Buy if probability > threshold
            - 'top_n': Buy top N% by probability
            - 'ensemble': Combine multiple model signals
        threshold : float
            Probability threshold for buy signals (default: 0.5)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals for each model
        """
        print(f"\nGenerating signals using '{strategy}' strategy...")
        
        # Create base signals dataframe
        signals_df = df[['date', 'ticker']].copy()
        
        # Add predictions and signals for each model
        for model_name, preds in predictions.items():
            # Add predictions
            signals_df[f'{model_name}_prediction'] = preds['labels']
            signals_df[f'{model_name}_probability'] = preds['probabilities']
            
            # Generate signals based on strategy
            if strategy == 'threshold':
                signals_df[f'{model_name}_signal'] = self._threshold_signal(
                    preds['probabilities'], threshold
                )
            elif strategy == 'top_n':
                signals_df[f'{model_name}_signal'] = self._top_n_signal(
                    preds['probabilities'], df['ticker'], top_pct=0.2
                )
            else:
                # Default to threshold
                signals_df[f'{model_name}_signal'] = self._threshold_signal(
                    preds['probabilities'], threshold
                )
            
            # Add signal descriptions
            signals_df[f'{model_name}_action'] = signals_df[f'{model_name}_signal'].map({
                1: 'BUY',
                0: 'HOLD',
                -1: 'SELL'
            })
        
        # Create ensemble signal (majority vote)
        if len(predictions) > 1:
            signal_cols = [f'{name}_signal' for name in predictions.keys()]
            signals_df['ensemble_signal'] = signals_df[signal_cols].mode(axis=1)[0]
            signals_df['ensemble_action'] = signals_df['ensemble_signal'].map({
                1: 'BUY',
                0: 'HOLD',
                -1: 'SELL'
            })
            
            # Ensemble confidence (agreement between models)
            signals_df['ensemble_confidence'] = signals_df[signal_cols].apply(
                lambda row: (row == row.mode()[0]).sum() / len(row), axis=1
            )
        
        self.signals_df = signals_df
        print(f"Generated signals for {len(signals_df)} samples")
        
        return signals_df
    
    def _threshold_signal(self, probabilities, threshold=0.5):
        """
        Generate signals based on probability threshold.
        
        Parameters:
        -----------
        probabilities : array-like
            Prediction probabilities
        threshold : float
            Probability threshold
            
        Returns:
        --------
        array
            Signals: 1 (BUY), 0 (HOLD), -1 (SELL)
        """
        signals = np.zeros(len(probabilities))
        signals[probabilities > threshold] = 1  # BUY
        signals[probabilities < (1 - threshold)] = -1  # SELL
        return signals.astype(int)
    
    def _top_n_signal(self, probabilities, tickers, top_pct=0.2):
        """
        Generate signals for top N% stocks by probability on each day.
        
        Parameters:
        -----------
        probabilities : array-like
            Prediction probabilities
        tickers : array-like
            Stock tickers
        top_pct : float
            Percentage of top stocks to buy (default: 20%)
            
        Returns:
        --------
        array
            Signals: 1 (BUY), 0 (HOLD), -1 (SELL)
        """
        signals = np.zeros(len(probabilities))
        
        # For each ticker, rank by probability
        df_temp = pd.DataFrame({
            'probability': probabilities,
            'ticker': tickers
        })
        
        # Mark top N% as BUY, bottom N% as SELL
        for ticker in df_temp['ticker'].unique():
            ticker_mask = df_temp['ticker'] == ticker
            ticker_probs = df_temp.loc[ticker_mask, 'probability']
            
            top_threshold = ticker_probs.quantile(1 - top_pct)
            bottom_threshold = ticker_probs.quantile(top_pct)
            
            signals[ticker_mask & (ticker_probs > top_threshold)] = 1
            signals[ticker_mask & (ticker_probs < bottom_threshold)] = -1
        
        return signals.astype(int)
    
    def analyze_signals(self):
        """
        Analyze generated signals and print statistics.
        
        Returns:
        --------
        dict
            Dictionary of signal statistics
        """
        if self.signals_df is None:
            print("No signals generated yet. Run create_signals() first.")
            return None
        
        print("\n" + "="*60)
        print("SIGNAL ANALYSIS")
        print("="*60)
        
        stats = {}
        
        # Get all signal columns
        signal_cols = [col for col in self.signals_df.columns if col.endswith('_signal')]
        
        for signal_col in signal_cols:
            model_name = signal_col.replace('_signal', '')
            
            signal_counts = self.signals_df[signal_col].value_counts()
            total = len(self.signals_df)
            
            stats[model_name] = {
                'buy_signals': signal_counts.get(1, 0),
                'hold_signals': signal_counts.get(0, 0),
                'sell_signals': signal_counts.get(-1, 0),
                'buy_pct': (signal_counts.get(1, 0) / total) * 100,
                'hold_pct': (signal_counts.get(0, 0) / total) * 100,
                'sell_pct': (signal_counts.get(-1, 0) / total) * 100
            }
            
            print(f"\n{model_name}:")
            print(f"  BUY:  {stats[model_name]['buy_signals']:4d} ({stats[model_name]['buy_pct']:.1f}%)")
            print(f"  HOLD: {stats[model_name]['hold_signals']:4d} ({stats[model_name]['hold_pct']:.1f}%)")
            print(f"  SELL: {stats[model_name]['sell_signals']:4d} ({stats[model_name]['sell_pct']:.1f}%)")
        
        return stats
    
    def plot_signal_distribution(self):
        """Plot distribution of signals across models."""
        if self.signals_df is None:
            print("No signals to plot. Run create_signals() first.")
            return
        
        signal_cols = [col for col in self.signals_df.columns if col.endswith('_signal')]
        
        if not signal_cols:
            print("No signal columns found.")
            return
        
        # Prepare data for plotting
        plot_data = []
        for col in signal_cols:
            model_name = col.replace('_signal', '')
            signal_counts = self.signals_df[col].value_counts()
            
            plot_data.append({
                'Model': model_name,
                'BUY': signal_counts.get(1, 0),
                'HOLD': signal_counts.get(0, 0),
                'SELL': signal_counts.get(-1, 0)
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df_plot))
        width = 0.6
        
        buy_bars = ax.bar(x, df_plot['BUY'], width, label='BUY', color='green', alpha=0.7)
        hold_bars = ax.bar(x, df_plot['HOLD'], width, bottom=df_plot['BUY'], 
                          label='HOLD', color='gray', alpha=0.7)
        sell_bars = ax.bar(x, df_plot['SELL'], width, 
                          bottom=df_plot['BUY'] + df_plot['HOLD'],
                          label='SELL', color='red', alpha=0.7)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Number of Signals', fontsize=12)
        ax.set_title('Signal Distribution by Model', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.outputs_dir}/visualizations/signal_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved signal distribution plot: {save_path}")
        plt.close()
    
    def plot_probability_distribution(self):
        """Plot distribution of prediction probabilities."""
        if self.signals_df is None:
            print("No signals to plot. Run create_signals() first.")
            return
        
        prob_cols = [col for col in self.signals_df.columns if col.endswith('_probability')]
        
        if not prob_cols:
            print("No probability columns found.")
            return
        
        fig, axes = plt.subplots(1, len(prob_cols), figsize=(5*len(prob_cols), 4))
        
        if len(prob_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(prob_cols):
            model_name = col.replace('_probability', '')
            
            axes[idx].hist(self.signals_df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
            axes[idx].set_xlabel('Probability', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{model_name}\nProbability Distribution', fontsize=11)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.outputs_dir}/visualizations/probability_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved probability distribution plot: {save_path}")
        plt.close()
    
    def save_signals(self, filename='signals.csv'):
        """
        Save signals to CSV file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        if self.signals_df is None:
            print("No signals to save. Run create_signals() first.")
            return
        
        save_path = f"{self.outputs_dir}/predictions/{filename}"
        self.signals_df.to_csv(save_path, index=False)
        print(f"\nSaved signals to: {save_path}")
        print(f"  Total signals: {len(self.signals_df)}")
        print(f"  Columns: {len(self.signals_df.columns)}")
        
        return save_path
    
    def get_latest_signals(self, n=10):
        """
        Get the most recent signals.
        
        Parameters:
        -----------
        n : int
            Number of recent signals to return
            
        Returns:
        --------
        pd.DataFrame
            Most recent signals
        """
        if self.signals_df is None:
            print("No signals generated yet.")
            return None
        
        # Convert date to datetime if it's not already
        df = self.signals_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date and get most recent
        df_sorted = df.sort_values('date', ascending=False)
        
        return df_sorted.head(n)
    
    def filter_signals_by_action(self, action='BUY', model_name=None):
        """
        Filter signals by action type.
        
        Parameters:
        -----------
        action : str
            Action to filter: 'BUY', 'SELL', or 'HOLD'
        model_name : str, optional
            Specific model to filter. If None, uses ensemble.
            
        Returns:
        --------
        pd.DataFrame
            Filtered signals
        """
        if self.signals_df is None:
            print("No signals generated yet.")
            return None
        
        if model_name:
            action_col = f'{model_name}_action'
        else:
            action_col = 'ensemble_action' if 'ensemble_action' in self.signals_df.columns else None
        
        if action_col is None or action_col not in self.signals_df.columns:
            print(f"Action column not found: {action_col}")
            return None
        
        filtered = self.signals_df[self.signals_df[action_col] == action]
        
        print(f"\nFound {len(filtered)} {action} signals")
        return filtered


def main():
    """
    Main execution function for signal generation.
    """
    print("="*60)
    print("SIGNAL GENERATION PIPELINE")
    print("="*60)
    
    # Initialize signal generator
    sg = SignalGenerator(
        models_dir='models',
        data_dir='data/processed',
        outputs_dir='outputs'
    )
    
    # Load models
    sg.load_models()
    
    # Load test data
    X_test, y_test, test_df = sg.load_data(data_type='test')
    
    # Generate predictions
    predictions = sg.generate_predictions(X_test)
    
    # Create signals using threshold strategy
    signals_df = sg.create_signals(
        predictions, 
        test_df, 
        strategy='threshold', 
        threshold=0.5
    )
    
    # Analyze signals
    stats = sg.analyze_signals()
    
    # Generate visualizations
    sg.plot_signal_distribution()
    sg.plot_probability_distribution()
    
    # Save signals
    sg.save_signals('signals.csv')
    
    # Show some example signals
    print("\n" + "="*60)
    print("EXAMPLE SIGNALS (Most Recent)")
    print("="*60)
    latest = sg.get_latest_signals(n=10)
    display_cols = ['date', 'ticker', 'ensemble_action', 'ensemble_confidence']
    if all(col in latest.columns for col in display_cols):
        print(latest[display_cols].to_string(index=False))
    
    print("\n" + "="*60)
    print("SIGNAL GENERATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review signals in outputs/predictions/signals.csv")
    print("2. Check visualizations in outputs/visualizations/")
    print("3. Run backtest.py to evaluate trading performance")
    
    return sg


if __name__ == "__main__":
    sg = main()