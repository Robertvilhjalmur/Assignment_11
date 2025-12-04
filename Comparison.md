# Model and Strategy Comparison

## Dataset Overview

* **Data** : 5 tech stocks (AAPL, MSFT, GOOG, TSLA, AMZN), Jan-Dec 2023
* **Final dataset** : 1,235 observations after removing NaN values
* **Split** : 985 training samples (Jan-Oct) | 250 test samples (Oct-Dec, 50 trading days)
* **Test period** : October-December 2023 was a down market

## Model Performance

### Classification Accuracy

| Model               | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | ------------- | --------- | ------ | -------- | ------- |
| Random Forest       | 53.2%         | 0.519     | 0.446  | 0.480    | 0.522   |
| Logistic Regression | 51.2%         | 0.492     | 0.256  | 0.337    | 0.497   |
| XGBoost             | 49.6%         | 0.481     | 0.512  | 0.496    | 0.488   |

All models performed barely better than random guessing (50%). ROC-AUC scores near 0.5 confirm weak predictive power.

### Overfitting Problem

| Model               | Train Accuracy | Test Accuracy | Gap    |
| ------------------- | -------------- | ------------- | ------ |
| XGBoost             | 80.0%          | 49.6%         | -30.4% |
| Random Forest       | 78.7%          | 53.2%         | -25.5% |
| Logistic Regression | 52.6%          | 51.2%         | -1.4%  |

Random Forest and XGBoost severely overfit. Cross-validation accuracy was even worse (40-45%).

## Feature Importance

| Feature                  | Importance Level | Description                         |
| ------------------------ | ---------------- | ----------------------------------- |
| return_1d                | High             | Yesterday's return (most important) |
| rsi_14                   | Medium           | RSI momentum indicator              |
| return_5d                | Medium           | 5-day return                        |
| macd                     | Low              | MACD indicator                      |
| return_3d, sma_5, sma_10 | Very Low         | Minimal impact                      |

Models learned to follow short-term momentum. Moving averages were largely ignored. See `outputs/visualizations/feature_importance_*.png`.

## Trading Performance

| Strategy                      | Total Return     | Sharpe Ratio    | Max Drawdown     | Trades | Win Rate | Profit Factor |
| ----------------------------- | ---------------- | --------------- | ---------------- | ------ | -------- | ------------- |
| **Logistic Regression** | **-0.10%** | **-0.30** | -0.90%           | 6      | 66.67%   | 7.29          |
| Random Forest                 | -0.20%           | -0.58           | -0.90%           | 46     | 52.17%   | 1.16          |
| Ensemble                      | -0.23%           | -0.57           | -0.97%           | 40     | 50.00%   | 1.13          |
| XGBoost                       | -0.72%           | -1.84           | -1.29%           | 55     | 43.64%   | 0.81          |
| **Buy-and-Hold**        | **-2.20%** | **-1.70** | **-3.85%** | 0      | N/A      | N/A           |

 **Key Finding** : All ML strategies beat buy-and-hold by 1.5-2.1%, losing much less in the down market.

### Signal Distribution

| Model               | BUY   | SELL  | Trading Style      |
| ------------------- | ----- | ----- | ------------------ |
| Logistic Regression | 25.2% | 74.8% | Very conservative  |
| Random Forest       | 41.6% | 58.4% | Moderately bearish |
| XGBoost             | 51.6% | 48.4% | Balanced           |
| Ensemble            | 41.2% | 58.8% | Moderately bearish |

See `outputs/visualizations/signal_distribution.png` for visual comparison.

## Best Model Analysis

| Criteria            | Winner                       | Reason                                      |
| ------------------- | ---------------------------- | ------------------------------------------- |
| Prediction Accuracy | Random Forest (53.2%)        | Highest test accuracy but overfit badly     |
| Trading Performance | Logistic Regression (-0.10%) | Smallest loss, highest profit factor (7.29) |
| Risk Management     | Logistic Regression (-0.90%) | Lowest drawdown                             |
| Balance             | Ensemble                     | 40 trades, 50% win rate, reliable           |

**Overall Winner: Logistic Regression**

* Made only 6 trades but had 66.67% win rate
* Conservative approach protected capital in down market
* Minimal overfitting (only 1.4% gap)
* Best risk-adjusted returns

## Visualizations

| File                          | Key Finding                                          |
| ----------------------------- | ---------------------------------------------------- |
| `equity_curves.png`         | ML strategies stayed flat, buy-and-hold declined     |
| `confusion_matrix_*.png`    | Heavy misclassification across all models            |
| `feature_importance_*.png`  | return_1d dominated, technical indicators minimal    |
| `drawdowns.png`             | ML limited losses to ~1%, buy-and-hold dropped 3.85% |
| `roc_curves_comparison.png` | All curves near diagonal (weak prediction)           |

All visualizations saved in `outputs/backtest_results/` and `outputs/visualizations/`.

## Limitations

1. **Barely better than guessing** : 50-53% accuracy is not reliable
2. **Severe overfitting** : Complex models failed to generalize
3. **Short test period** : Only 50 trading days, not statistically robust
4. **Down market only** : Results may not hold in bull markets
5. **No transaction costs** : Real returns would be 1-2% lower
6. **Momentum strategy** : Models only learned to follow recent trends

## Conclusions

Models achieved 50-53% accuracy (barely above random). Despite poor prediction, all ML strategies beat buy-and-hold benchmark by protecting capital in a down market. That is mostly because GOOGL had a huge drop in the period.

**Key Insights:**

* Low accuracy doesn't mean unprofitable (Logistic Regression: 51% accuracy, best returns)
* Risk management matters more than prediction accuracy
* Simple models can outperform complex ones
* Overfitting is a major problem in financial ML
* Markets are hard to predict (efficient market hypothesis confirmed)

This project demonstrates ML pipeline implementation but highlights the difficulty of financial forecasting. Not suitable for real trading.
