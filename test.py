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

# Load trained models from TRAINED_MODEL_DIR
trained_models = {}
model_paths = [
    "A2C_10k_126.zip",
    "A2C_10k_189.zip",
    "A2C_10k_252.zip",
    "A2C_5k_126.zip",
    "A2C_5k_189.zip",
    "A2C_5k_252.zip",
    "DDPG_10k_126.zip",
    "DDPG_10k_189.zip",
    "DDPG_10k_252.zip",
    "DDPG_5k_126.zip",
    "DDPG_5k_189.zip",
    "DDPG_5k_252.zip",
    "PPO_10k_126.zip",
    "PPO_10k_189.zip",
    "PPO_5k_126.zip",
    "PPO_5k_189.zip",
    "SAC_10k_126.zip",
    "SAC_10k_189.zip",
    "SAC_5k_126.zip",
    "SAC_5k_189.zip",
    "TD3_10k_126.zip",
    "TD3_10k_189.zip",
    "TD3_5k_126.zip",
    "TD3_5k_189.zip"
]

from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3

for model_path in model_paths:
    model_name = model_path.split('_')[0]  # Extract model name from the file name
    model = None
    
    if model_name == 'A2C':
        model = A2C.load(f"{TRAINED_MODEL_DIR}/{model_path}")
    elif model_name == 'PPO':
        model = PPO.load(f"{TRAINED_MODEL_DIR}/{model_path}")
    elif model_name == 'DDPG':
        model = DDPG.load(f"{TRAINED_MODEL_DIR}/{model_path}")
    elif model_name == 'SAC':
        model = SAC.load(f"{TRAINED_MODEL_DIR}/{model_path}")
    elif model_name == 'TD3':
        model = TD3.load(f"{TRAINED_MODEL_DIR}/{model_path}")
    
    if model:
        trained_models[model_path] = model
    else:
        print(f"Failed to load model from {model_path}")


def predict_next_day_price(model, data, date):
    try:
        # Filter data for the specific date
        features_for_prediction = data[data['date'] == date].drop(columns=['date'])
        
        # Debugging: Print shape and content of features_for_prediction
        print(f"Shape of features_for_prediction: {features_for_prediction.shape}")
        print(f"Features for prediction:\n{features_for_prediction.head()}")
        
        # Ensure features_for_prediction has the correct shape and contains data
        if features_for_prediction.empty:
            raise ValueError(f"No data found for date {date} in processed data.")
        
        # Predict using the model
        prediction = model.predict(features_for_prediction)
        
        return prediction
    
    except Exception as e:
        print(f"Error predicting for date {date}: {e}")
        return None

# Example usage:
date_to_predict = '2024-08-08'
model_to_use = trained_models['TD3_10k_126.zip']  # Example: Using A2C model
prediction = predict_next_day_price(model_to_use, processed, date_to_predict)
if prediction is not None:
    print(f"Predicted price for {date_to_predict}: {prediction}")
else:
    print(f"No prediction made.")
