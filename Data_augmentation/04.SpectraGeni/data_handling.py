# data_handling.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_loading_all_parquet(crop_names):
    all_dfs = []
    for i, crop_name in enumerate(crop_names):
        try:
            # Load real data from Parquet
            filename_real = f"../Data/Train_80/Parquet/{crop_name}.parquet"
            df_real = pd.read_parquet(filename_real)
            
            # Load synthetic data from BSMOTE from Parquet
#             filename_syn = f"../Data/Syn_BSMOTE/{crop_name}.parquet"
#             df_syn = pd.read_parquet(filename_syn)  

            # Combine real and synthetic for one crop
            df_combined = pd.concat([df_real], ignore_index=True)
#             df_combined = pd.concat([df_real, df_syn], ignore_index=True)
            
            # Add the numerical label for the crop
            df_combined['label'] = i  # Assign integer label
            
            all_dfs.append(df_combined)
            print(f"Loaded {crop_name} (label {i}). Shape: {df_combined.shape}")

        except FileNotFoundError as e:
            print(f"ERROR: Could not find Parquet file for {crop_name}.")
            print("Please run the `convert_all_excel_to_parquet.py` script first.")
            print(f"Details: {e}")
            return None

    # Concatenate all crop dataframes into one
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined DataFrame shape: {final_df.shape}")
    print("Value counts for labels:\n", final_df['label'].value_counts())

    return final_df


def data_loading_all_excel(crop_names):
    """
    Loads and combines data for all specified crops into a single DataFrame.
    """
    all_dfs = []
    for i, crop_name in enumerate(crop_names):
        # Load real data
        filename_real = f"../Data/{crop_name}.xlsx"
        df_real = pd.read_excel(filename_real, sheet_name='Sheet1')
        
        # Load synthetic data from BSMOTE
        filename_syn = f"../Data/Syn_BSMOTE/{crop_name}.xlsx"
        df_syn = pd.read_excel(filename_syn, sheet_name='Sheet1')
        
        # Combine real and synthetic for one crop
        df_combined = pd.concat([df_real, df_syn], ignore_index=True)
        
        # Add the numerical label for the crop
        df_combined['label'] = i
        
        all_dfs.append(df_combined)
        print(f"Loaded {crop_name} (label {i}). Shape: {df_combined.shape}")

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined DataFrame shape: {final_df.shape}")
    print("Value counts for labels:\n", final_df['label'].value_counts())
    
    return final_df


def data_prep_conditional(df):
    """
    Prepares data and keeps labels for CVAE.
    """
    # Separate features (X) and labels (y)
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return scaler, train_loader, test_loader


def save_data(synthetic_data, crop_name, bands, label_id):
    """
    Saves generated data to a Parquet file.
    """
    bands = [str(b) for b in bands]
    
    synthetic_df = pd.DataFrame(synthetic_data, columns=bands)

    synthetic_df['label'] = label_id  
    # >>> FIX: label should be numeric, not crop_name <<<

    filename = f"../Results/CVAE_generated/{crop_name}.parquet"
    synthetic_df.to_parquet(
        filename,
        engine='pyarrow',
        index=False,
        compression='snappy'
    )
    
    print(f"Saved {synthetic_data.shape[0]} synthetic samples for {crop_name} (label {label_id})")


def convert_all_excel_to_parquet():
    print("--- Starting Automatic Excel to Parquet Conversion ---")

    directories_to_scan = [
        "../Data/",
        "../Data/Syn_BSMOTE/"
    ]

    for directory_path in directories_to_scan:
        print(f"\n--- Scanning directory: {directory_path} ---")
        
        excel_files = glob.glob(os.path.join(directory_path, '*.xlsx'))
        
        if not excel_files:
            print("  No Excel files found in this directory.")
            continue

        for excel_path in excel_files:
            parquet_path = excel_path.replace('.xlsx', '.parquet')
            
            try:
                print(f"  Processing: {os.path.basename(excel_path)}")
                
                df = pd.read_excel(excel_path)

                rename_map = {col: str(col) for col in df.columns}
                df.rename(columns=rename_map, inplace=True)
                
                df.to_parquet(
                    parquet_path,
                    engine='pyarrow',
                    index=False,
                    compression='snappy'
                )
                
                print(f"  -> Saved to: {os.path.basename(parquet_path)}")
            
            except Exception as e:
                print(f"  -> ERROR: Failed to convert {os.path.basename(excel_path)}: {e}")

    print("\n--- Conversion process complete. ---")
