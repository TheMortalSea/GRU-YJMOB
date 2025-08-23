import pandas as pd
import sys
import os

def check_for_mask(filepath):
    """
    Checks if a dataset has the '999' mask value in its 'x' or 'y' columns.
    It specifically targets the last 3000 unique users, which represent the test set.
    
    Args:
        filepath (str): The path to the .csv or .csv.gz file.
    """
    try:
        # Load the specified CSV file, handling gzip compression
        compression = 'gzip' if filepath.endswith('.gz') else None
        df = pd.read_csv(filepath, compression=compression)

        # Get the unique user IDs
        all_uids = df['uid'].unique()
        
        # Check if there are at least 3000 users
        if len(all_uids) < 3000:
            print("Error: The dataset contains fewer than 3000 unique users. Cannot identify a test set.")
            return

        # Get the UIDs for the test set (the last 3000)
        test_uids = all_uids[-3000:]
        
        # Filter the DataFrame to only include the test set users
        test_df = df[df['uid'].isin(test_uids)]

        # Check if the 'x' and 'y' columns exist
        if 'x' not in test_df.columns or 'y' not in test_df.columns:
            print("Error: The CSV file must contain 'x' and 'y' columns.")
            return

        # Check for the presence of the mask value (999) in the filtered test set
        has_mask_x = (test_df['x'] == 999).any()
        has_mask_y = (test_df['y'] == 999).any()

        if has_mask_x or has_mask_y:
            print(f"\n✅ Found '999' mask value for the last 3000 test users.")
            print("This dataset is correctly formatted for testing.")
        else:
            print(f"\n❌ Did not find '999' mask value for the last 3000 test users.")
            print("This indicates the test set may have already been processed.")

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'. Please check the path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Hardcoded filepath as requested
    check_for_mask('./cityD-dataset.csv.gz')

