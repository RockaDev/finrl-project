# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import sys
sys.path.append("../FinRL-Library")

# Create necessary folders and import configurations
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# Download data using YahooDownloader
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TEST_START_DATE = '2021-10-01'
TEST_END_DATE = '2023-03-01'

df = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=['GME']).fetch_data()

# Preprocess data
INDICATORS = ['macd', 'rsi_30', 'cci_30', 'dx_30']
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
    user_defined_feature=False
)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf, 0)
print("Successfully added technical indicators and turbulence index")
print(processed.head())

# Design Environment
stock_dimension = len(processed.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5
}

# Implement DRL Algorithms
rebalance_window = 63
validation_window = 63

ensemble_agent = DRLEnsembleAgent(
    df=processed,
    train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
    val_test_period=(TEST_START_DATE, TEST_END_DATE),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs
)

A2C_model_kwargs = {
    'n_steps': 5,
    'ent_coef': 0.005,
    'learning_rate': 0.0007
}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128
}

DDPG_model_kwargs = {
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 64
}

SAC_model_kwargs = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

TD3_model_kwargs = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.0001
}

timesteps_dict = {
    'a2c': 10_000,
    'ppo': 10_000,
    'ddpg': 10_000,
    'sac': 10_000,
    'td3': 10_000
}

df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs,
    PPO_model_kwargs,
    DDPG_model_kwargs,
    SAC_model_kwargs,
    TD3_model_kwargs,
    timesteps_dict
)

# Extract trained models
trained_models = ensemble_agent.get_trained_models()

# Example: Use trained models for prediction
def predict_next_day_price(model, data, date):
    # Assume 'model' is a trained RL model
    # 'data' should contain features necessary for prediction
    # 'date' is the date for which prediction is needed
    
    # Example: Retrieve features for the given date
    features_for_prediction = data[data['date'] == date].drop(columns=['date'])
    
    # Example: Use the model to predict next day's price
    prediction = model.predict(features_for_prediction)
    
    return prediction

# Example usage:
date_to_predict = '2023-03-02'
model_to_use = trained_models['a2c']  # Example: Using A2C model
prediction = predict_next_day_price(model_to_use, processed, date_to_predict)
print(f"Predicted price for {date_to_predict}: {prediction}")

# Optionally, you can further visualize or analyze the predictions or other outputs as needed.
