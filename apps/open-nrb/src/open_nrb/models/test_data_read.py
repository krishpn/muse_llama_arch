#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
# ]
# ///

import pandas as pd
from pathlib import Path

# Get the absolute path of the current script (inside models/)
current_file = Path(__file__).resolve()
print(f"Script path: {current_file}")

# Go up two levels from models/ to open_nrb/, then into data/processed/
file_path = current_file.parent.parent / 'data' / 'processed' / 'monthly_statistical_bfis' /'nrb_balance_sheet.csv'

for file in file_path.parent.glob('*.csv'):
    # Read CSV
    df = pd.read_csv(file)
    print("==" * 50)

    if 'year_bs' in df.columns:
        # Convert to numeric safely (coercing any bad strings to NaN, then dropping or filling)
        df['year_bs'] = pd.to_numeric(df['year_bs'], errors='coerce')
        
        # Sort the *entire dataframe* by year_bs
        df = df.sort_values(by='year_bs', ascending=True)
        
        # Get sorted unique years (using sorted() on the unique array ensures chronological order)
        unique_years = sorted([int(y) for y in df['year_bs'].dropna().unique()])
        
        print(f"File: {file.name}")
        print(f"Shape: {df.shape}")
        print(f"Sorted Years (BS): {unique_years}")
        print(f"Total Unique Years: {len(unique_years)}")
        print(f"Columns:\n{df.columns}")
        if 'line_item' in df.columns:   
            print(f"Line Items:\n{df['line_item'].unique().to_numpy()}")
        # print(f"First 5 rows:\n{df.head(10)}")