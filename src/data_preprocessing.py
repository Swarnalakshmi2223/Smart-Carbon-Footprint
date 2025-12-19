import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Load the carbon footprint dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\nChecking for missing values...")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    
    if missing_counts.sum() > 0:
        print("\nHandling missing values...")
        # For numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  - Filled {col} with median")
        
        # For categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  - Filled {col} with mode")
    else:
        print("No missing values found.")
    
    return df

def encode_categorical_features(df):
    """Encode categorical features using Label Encoding."""
    print("\nEncoding categorical features...")
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # Encode diet_type
    label_encoder = LabelEncoder()
    df_encoded['diet_type_encoded'] = label_encoder.fit_transform(df_encoded['diet_type'])
    
    # Store the mapping for reference
    diet_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"  - diet_type encoded: {diet_mapping}")
    
    # Drop original categorical column
    df_encoded = df_encoded.drop('diet_type', axis=1)
    
    return df_encoded, label_encoder

def normalize_features(X_train, X_test):
    """Normalize numerical features using StandardScaler."""
    print("\nNormalizing numerical features...")
    
    scaler = StandardScaler()
    
    # Fit on training data and transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  - Features normalized (mean=0, std=1)")
    
    return X_train_scaled, X_test_scaled, scaler

def split_data(df, target_column='carbon_footprint_kg_co2', test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"\nSplitting data into train and test sets ({int((1-test_size)*100)}:{int(test_size*100)})...")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='../data'):
    """Save preprocessed data to CSV files."""
    print(f"\nSaving preprocessed data to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert arrays back to DataFrame for saving
    if isinstance(X_train, np.ndarray):
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
    
    # Save to CSV
    X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=['carbon_footprint_kg_co2'])
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=['carbon_footprint_kg_co2'])
    
    print("  - X_train.csv")
    print("  - X_test.csv")
    print("  - y_train.csv")
    print("  - y_test.csv")
    print("\nPreprocessing completed successfully!")

def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("Carbon Footprint Data Preprocessing Pipeline")
    print("="*60)
    
    # Load data
    data_path = '../data/carbon_footprint.csv'
    df = load_data(data_path)
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df_encoded, label_encoder = encode_categorical_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Save preprocessed data
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  - Original dataset shape: {df.shape}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Training samples: {X_train_scaled.shape[0]}")
    print(f"  - Test samples: {X_test_scaled.shape[0]}")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, label_encoder = main()
