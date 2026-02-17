import sys
import os

# --- PATH FIX: Add Project Root to System Path ---
# This allows 'python src/modeling.py' to find modules like 'src.database'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import pickle

# Updated Imports using 'src.' prefix
from src.database import AQIDatabase
from src.preprocessing import preprocess_data

def train_and_evaluate():
    """
    Trains multiple models and selects the best one based on RMSE.
    """
    print("Fetching data from MongoDB...")
    db = AQIDatabase()
    df = db.fetch_data()
    
    if df.empty:
        print("No data found in database. Run data_ingestion.py first.")
        return

    print(f"Data fetched: {len(df)} records.")
    
    # Preprocess
    df = preprocess_data(df, is_training=True)
    
    # Shift target logic
    df['target'] = df['us_aqi'].shift(-1)
    df = df.dropna()
    
    # Feature selection
    features = [c for c in df.columns if 'lag' in c or 'rolling' in c or c in ['hour', 'day_of_week', 'month']]
    print(f"Features used: {features}")
    
    X = df[features]
    y = df['target']
    
    # Time-based train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    best_model = None
    best_score = float('inf')
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_name = name
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            
    if best_model is None:
        print("\nError: No models were successfully trained.")
        sys.exit(1)

    print(f"\nBest model: {best_name} with RMSE: {best_score:.4f}")
    
    # --- SAVE TO 'models' FOLDER ---
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(os.path.join(models_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
        
    print(f"Model and features saved to {models_dir}")
    return best_model, features

def predict_next_72_hours(model, features, recent_data):
    predictions = []
    history_df = recent_data.tail(100).copy()
    max_physical_aqi = 500
    
    for i in range(72):
        df_processed = preprocess_data(history_df.copy(), is_training=False)
        input_row = df_processed.iloc[-1:]
        X_input = input_row[features]
        
        raw_pred = model.predict(X_input)[0]
        prev_aqi = history_df['us_aqi'].iloc[-1]
        
        dampening = 0.15
        lower_bound = prev_aqi * (1 - dampening)
        upper_bound = prev_aqi * (1 + dampening)
        
        pred_aqi = np.clip(raw_pred, lower_bound, upper_bound)
        pred_aqi = np.clip(pred_aqi, 0, max_physical_aqi)
        
        last_date = history_df['date'].iloc[-1]
        next_date = last_date + pd.Timedelta(hours=1)
        
        new_row = pd.DataFrame({'date': [next_date], 'us_aqi': [pred_aqi]})
        last_known = history_df.iloc[-1].to_dict()
        for col, val in last_known.items():
            if col not in ['date', 'us_aqi', 'target'] and col not in new_row.columns and 'lag' not in col and 'rolling' not in col:
                 new_row[col] = val
                 
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        predictions.append({'date': next_date, 'predicted_aqi': float(pred_aqi)})
        
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    train_and_evaluate()