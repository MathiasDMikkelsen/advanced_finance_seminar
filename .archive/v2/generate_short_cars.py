import pandas as pd
import numpy as np

def compute_cars():
    print("Loading daily returns...")
    df_daily = pd.read_csv('data/combined/all_daily_returns.csv', low_memory=False)
    
    # Collect results
    results = []
    
    # Identify the start indices for all 1000 events
    col_names = df_daily.columns.tolist()
    event_indices = [i for i, col in enumerate(col_names) if '|' in str(col)]
    
    # Iterate over every event block
    for i in event_indices:
        col_name = col_names[i]
        event_id = str(col_name).strip()
        
        # Trd_Day is in column i+4, Return is in column i+1
        trd_day_col = df_daily.iloc[:, i+4]
        ret_col = df_daily.iloc[:, i+1]
        
        # Filter for the relevant rows (drop headers/NaNs)
        # We expect Trd_Day to be integers represented as strings
        valid_mask = trd_day_col.astype(str).str.replace('-', '').str.isnumeric()
        
        trd_day = pd.to_numeric(trd_day_col[valid_mask])
        ret = pd.to_numeric(ret_col[valid_mask]) / 100.0  # Returns are in percentages
        
        df_event = pd.DataFrame({'Trd_Day': trd_day, 'Return': ret}).dropna()
        
        # Calculate requested CARs
        car_2_3 = df_event[(df_event['Trd_Day'] >= 2) & (df_event['Trd_Day'] <= 3)]['Return'].sum()
        car_2_5 = df_event[(df_event['Trd_Day'] >= 2) & (df_event['Trd_Day'] <= 5)]['Return'].sum()
        car_2_7 = df_event[(df_event['Trd_Day'] >= 2) & (df_event['Trd_Day'] <= 7)]['Return'].sum()
        car_2_10 = df_event[(df_event['Trd_Day'] >= 2) & (df_event['Trd_Day'] <= 10)]['Return'].sum()
        car_2_15 = df_event[(df_event['Trd_Day'] >= 2) & (df_event['Trd_Day'] <= 15)]['Return'].sum()
        
        results.append({
            'Event_ID': event_id,
            'CAR_2_3': car_2_3,
            'CAR_2_5': car_2_5,
            'CAR_2_7': car_2_7,
            'CAR_2_10': car_2_10,
            'CAR_2_15': car_2_15
        })
        
    df_cars = pd.DataFrame(results)
    
    print("Loading all_relevant_data.csv...")
    df_main = pd.read_csv('data/combined/all_relevant_data.csv')
    
    # Merge df_cars into df_main
    df_merged = pd.merge(df_main, df_cars, on='Event_ID', how='left', suffixes=('', '_new'))
    
    # If columns existed, replace them
    for col in ['CAR_2_3', 'CAR_2_5', 'CAR_2_7', 'CAR_2_10', 'CAR_2_15']:
        if f"{col}_new" in df_merged.columns:
            df_merged[col] = df_merged[f"{col}_new"]
            df_merged.drop(columns=[f"{col}_new"], inplace=True)
            
    df_merged.to_csv('data/combined/all_relevant_data.csv', index=False)
    print(f"Updated relevant data shape: {df_merged.shape}")

if __name__ == '__main__':
    compute_cars()