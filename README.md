# chaitanya
import pandas as pd
import numpy as np
 
# Step 1: Load and preprocess data
def load_data(file_path):
   """Load PM2.5 monitoring data from a CSV file and parse timestamps."""
   df = pd.read_csv(file_path, parse_dates=['timestamp'], dayfirst=True)
   df.sort_values(by='timestamp', inplace=True)
   return df
 
# Step 2: Calculate median and differentiate signals
def classify_signals(df, pm_column='pm2.5'):
   """Classify signals into background (smooth) and fast signals based on median."""
   median = df[pm_column].median()
   df['background'] = df[pm_column] <= median
   df['fast_signal'] = df[pm_column] > median
   return df, median
 
# Step 3: Calculate local indoor concentrations
def calculate_local_indoor(df, pm_column='pm2.5'):
   """Calculate local indoor PM2.5 concentrations."""
   background_std = df.loc[df['background'], pm_column].std()
   df['local_indoor'] = np.nan
 
   for t in range(1, len(df)):
       df.loc[df.index[t], 'local_indoor'] = df.loc[df.index[t-1], pm_column] + 3 * background_std
 
   return df
 
# Step 4: Calculate local outdoor concentrations
def calculate_local_outdoor(df, pm_column='pm2.5'):
   """Calculate local outdoor PM2.5 concentrations using moving average subtraction method."""
   # Exclude fast signal values (Step 3.1)
   filtered_df = df[~df['fast_signal']].copy()
 
   # Helper function to calculate moving average for a given window size
   def moving_average_filter(data, window):
       return data.rolling(window=window, center=True, min_periods=1).mean()
 
   # Apply the hierarchical averaging
   for window in [60, 30, 15]:
       hourly_avg = moving_average_filter(filtered_df[pm_column], window)
 
       filtered_df[pm_column] = np.where(
           hourly_avg < filtered_df[pm_column],
           hourly_avg,
           filtered_df[pm_column]
       )
 
   filtered_df['local_outdoor'] = filtered_df[pm_column]
 
   # Merge back the calculated local outdoor concentrations
   df = df.merge(filtered_df[['timestamp', 'local_outdoor']], on='timestamp', how='left')
   return df
 
# Main Function
def process_pm25_data(file_path, pm_column='pm2.5'):
   """Process PM2.5 data to classify signals and calculate local indoor and outdoor concentrations."""
   df = load_data(file_path)
   df, median = classify_signals(df, pm_column)
   df = calculate_local_indoor(df, pm_column)
   df = calculate_local_outdoor(df, pm_column)
   return df
 
# Example usage
# Replace 'your_file.csv' with the actual file path to your data
file_path = 'D2_5Min.csv'  # Update with the correct path to your CSV file
processed_data = process_pm25_data(file_path, pm_column='pm2.5')
 
# Save or inspect the processed data
processed_data.to_csv('processed_pm25_data.csv', index=False)
print(processed_data.head())

