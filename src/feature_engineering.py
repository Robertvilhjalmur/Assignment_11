import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    A class to handle feature engineering for financial time series data.
    """
    
    def __init__(self, data_path='data/raw/market_data_ml.csv', 
                 config_path='data/raw/features_config.json'):
        """
        Initialize the FeatureEngineer.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw market data CSV file
        config_path : str
            Path to the features configuration JSON file
        """
        self.data_path = data_path
        self.config_path = config_path
        self.df = None
        self.features_config = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load raw market data from CSV."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date']).reset_index(drop=True)
        print(f"Loaded {len(self.df)} rows for {self.df['ticker'].nunique()} tickers")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        return self.df
    
    def load_config(self):
        """Load feature configuration from JSON."""
        print(f"\nLoading configuration from {self.config_path}...")
        with open(self.config_path, 'r') as f:
            self.features_config = json.load(f)
        print(f"Features to generate: {self.features_config['features']}")
        print(f"Target label: {self.features_config['label']}")
        return self.features_config
    
    def calculate_returns(self, df):
        """
        Calculate return features for different time periods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added return columns
        """
        print("\nCalculating returns...")
        
        # Calculate daily returns
        df['return_1d'] = df.groupby('ticker')['close'].pct_change(1)
        df['return_3d'] = df.groupby('ticker')['close'].pct_change(3)
        df['return_5d'] = df.groupby('ticker')['close'].pct_change(5)
        
        # Also calculate log returns (useful for some analysis)
        df['log_return_1d'] = df.groupby('ticker')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        
        return df
    
    def calculate_sma(self, df, windows=[5, 10]):
        """
        Calculate Simple Moving Averages.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        windows : list
            List of window sizes for SMA calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added SMA columns
        """
        print(f"Calculating SMAs with windows: {windows}...")
        
        for window in windows:
            col_name = f'sma_{window}'
            df[col_name] = df.groupby('ticker')['close'].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
            
            # Create ratio features (price relative to SMA)
            df[f'price_to_{col_name}'] = df['close'] / df[col_name]
        
        return df
    
    def calculate_rsi(self, df, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        period : int
            Period for RSI calculation (default: 14)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added RSI column
        """
        print(f"Calculating RSI with period {period}...")
        
        def compute_rsi(group):
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14'] = df.groupby('ticker', group_keys=False).apply(compute_rsi).values
        
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = 12-day EMA - 26-day EMA
        Signal Line = 9-day EMA of MACD Line
        MACD Histogram = MACD Line - Signal Line
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        fast : int
            Fast EMA period (default: 12)
        slow : int
            Slow EMA period (default: 26)
        signal : int
            Signal line EMA period (default: 9)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added MACD columns
        """
        print(f"Calculating MACD ({fast}/{slow}/{signal})...")
        
        def compute_macd(group):
            exp1 = group['close'].ewm(span=fast, adjust=False).mean()
            exp2 = group['close'].ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return pd.DataFrame({
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            })
        
        macd_df = df.groupby('ticker', group_keys=False).apply(compute_macd)
        df['macd'] = macd_df['macd'].values
        df['macd_signal'] = macd_df['macd_signal'].values
        df['macd_histogram'] = macd_df['macd_histogram'].values
        
        return df
    
    def create_labels(self, df):
        """
        Create target labels for classification.
        
        Direction: 1 if next day return > 0, else 0
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with return data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added label column
        """
        print("\nCreating target labels...")
        
        # Calculate next day's return
        df['next_day_return'] = df.groupby('ticker')['close'].pct_change(1).shift(-1)
        
        # Create binary direction label
        df['direction'] = (df['next_day_return'] > 0).astype(int)
        
        # Also create magnitude label for regression tasks (optional)
        df['next_day_return_label'] = df['next_day_return']
        
        print(f"Label distribution:\n{df['direction'].value_counts()}")
        print(f"Class balance: {df['direction'].value_counts(normalize=True)}")
        
        return df
    
    def engineer_features(self):
        """
        Main method to orchestrate all feature engineering steps.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all engineered features
        """
        # Load data and config
        self.load_data()
        self.load_config()
        
        # Calculate all features
        self.df = self.calculate_returns(self.df)
        self.df = self.calculate_sma(self.df, windows=[5, 10])
        self.df = self.calculate_rsi(self.df, period=14)
        self.df = self.calculate_macd(self.df)
        
        # Create labels
        self.df = self.create_labels(self.df)
        
        # Handle NaN values
        print(f"\nNaN values before cleaning:")
        print(self.df[self.features_config['features'] + ['direction']].isna().sum())
        
        # Drop rows with NaN values (caused by rolling calculations)
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=self.features_config['features'] + ['direction'])
        final_rows = len(self.df)
        print(f"\nDropped {initial_rows - final_rows} rows with NaN values")
        print(f"Final dataset: {final_rows} rows")
        
        return self.df
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame, optional
            Test features
            
        Returns:
        --------
        tuple
            Scaled training and test features (if test provided)
        """
        print("\nScaling features...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(
                X_test_scaled, 
                columns=X_test.columns, 
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, test_size=0.2):
        """
        Split data into train/test sets using time-based split.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing (default: 0.2)
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, train_df, test_df)
        """
        print(f"\nSplitting data with {test_size*100}% test size...")
        
        # Get feature columns from config
        feature_cols = self.features_config['features']
        label_col = self.features_config['label']
        
        # Time-based split for each ticker
        train_dfs = []
        test_dfs = []
        
        for ticker in self.df['ticker'].unique():
            ticker_df = self.df[self.df['ticker'] == ticker].copy()
            
            # Calculate split index (80/20 time-based)
            split_idx = int(len(ticker_df) * (1 - test_size))
            
            train_df = ticker_df.iloc[:split_idx]
            test_df = ticker_df.iloc[split_idx:]
            
            train_dfs.append(train_df)
            test_dfs.append(test_df)
            
            print(f"{ticker}: {len(train_df)} train, {len(test_df)} test samples")
        
        # Combine all tickers
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        # Extract features and labels
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[label_col]
        y_test = test_df[label_col]
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"\nFinal split:")
        print(f"Training: {len(X_train_scaled)} samples")
        print(f"Testing: {len(X_test_scaled)} samples")
        print(f"Training date range: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Testing date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, train_df, test_df
    
    def save_processed_data(self, train_df, test_df, output_dir='data/processed'):
        """
        Save processed features to CSV files.
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data
        test_df : pd.DataFrame
            Test data
        output_dir : str
            Directory to save processed data
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save train and test data
        train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
        
        # Save feature columns and scaler info
        feature_info = {
            'feature_columns': self.features_config['features'],
            'label_column': self.features_config['label'],
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist()
        }
        
        with open(f"{output_dir}/feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=4)
        
        print(f"Saved train_data.csv ({len(train_df)} rows)")
        print(f"Saved test_data.csv ({len(test_df)} rows)")
        print(f"Saved feature_info.json")


def main():
    """
    Main execution function for feature engineering.
    """
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Initialize feature engineer
    fe = FeatureEngineer(
        data_path='data/raw/market_data_ml.csv',
        config_path='data/raw/features_config.json'
    )
    
    # Engineer all features
    df_features = fe.engineer_features()
    
    # Split data
    X_train, X_test, y_train, y_test, train_df, test_df = fe.split_data(test_size=0.2)
    
    # Save processed data
    fe.save_processed_data(train_df, test_df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the processed data in data/processed/")
    print("2. Run train_model.py to train ML models")
    
    return fe, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    fe, X_train, X_test, y_train, y_test = main()