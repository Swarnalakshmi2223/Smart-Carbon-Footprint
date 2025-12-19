import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

def load_preprocessed_data(data_dir='../data'):
    """Load the preprocessed training and test data."""
    print("Loading preprocessed data...")
    
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance on both train and test sets."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Display results
    print("\nTraining Set Performance:")
    print(f"  - MAE:  {train_mae:.4f} kg CO₂")
    print(f"  - RMSE: {train_rmse:.4f} kg CO₂")
    print(f"  - R² Score: {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  - MAE:  {test_mae:.4f} kg CO₂")
    print(f"  - RMSE: {test_rmse:.4f} kg CO₂")
    print(f"  - R² Score: {test_r2:.4f}")
    
    # Calculate error percentage
    mean_carbon = np.mean(y_test)
    error_pct = (test_mae / mean_carbon) * 100
    print(f"\nMean Absolute Error: {error_pct:.2f}% of average carbon footprint")
    
    return {
        'model_name': model_name,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'predictions': y_test_pred
    }

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    print("\nTraining Linear Regression model...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("  ✓ Linear Regression training completed")
    
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest Regressor model."""
    print("\nTraining Random Forest Regressor model...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("  ✓ Random Forest training completed")
    print(f"  - Number of estimators: {model.n_estimators}")
    print(f"  - Max depth: {model.max_depth}")
    
    return model

def display_feature_importance(model, feature_names, top_n=5):
    """Display feature importance for Random Forest model."""
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importance (Top Features):")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(top_n, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

def compare_models(results_list):
    """Compare model performances and select the best one."""
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame(results_list)
    
    print("\nTest Set Metrics Comparison:")
    print("-" * 60)
    for _, row in comparison_df.iterrows():
        print(f"\n{row['model_name']}:")
        print(f"  MAE:  {row['test_mae']:.4f}")
        print(f"  RMSE: {row['test_rmse']:.4f}")
        print(f"  R² Score: {row['test_r2']:.4f}")
    
    # Select best model based on R² score
    best_idx = comparison_df['test_r2'].idxmax()
    best_model_name = comparison_df.loc[best_idx, 'model_name']
    best_r2 = comparison_df.loc[best_idx, 'test_r2']
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name} (R² = {best_r2:.4f})")
    print(f"{'='*60}")
    
    return best_model_name, comparison_df

def save_model(model, model_name, output_dir='../models'):
    """Save the trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'carbon_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"  - Model type: {model_name}")
    print(f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model_path

def save_model_metrics(results_df, output_dir='../models'):
    """Save model comparison metrics to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, 'model_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    
    print(f"\n✓ Model metrics saved to: {metrics_path}")

def main():
    """Main training pipeline."""
    print("="*60)
    print("Carbon Footprint Prediction - Model Training Pipeline")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Convert to numpy arrays for training
    X_train_array = X_train.values
    X_test_array = X_test.values
    
    # Train Linear Regression
    lr_model = train_linear_regression(X_train_array, y_train)
    lr_results = evaluate_model(lr_model, X_train_array, X_test_array, 
                                y_train, y_test, "Linear Regression")
    
    # Train Random Forest
    rf_model = train_random_forest(X_train_array, y_train)
    rf_results = evaluate_model(rf_model, X_train_array, X_test_array, 
                                y_train, y_test, "Random Forest Regressor")
    
    # Display feature importance for Random Forest
    display_feature_importance(rf_model, feature_names)
    
    # Compare models
    results_list = [
        {k: v for k, v in lr_results.items() if k != 'predictions'},
        {k: v for k, v in rf_results.items() if k != 'predictions'}
    ]
    
    best_model_name, comparison_df = compare_models(results_list)
    
    # Save best model
    if best_model_name == "Random Forest Regressor":
        best_model = rf_model
    else:
        best_model = lr_model
    
    model_path = save_model(best_model, best_model_name)
    
    # Save metrics
    save_model_metrics(comparison_df)
    
    # Final summary
    print("\n" + "="*60)
    print("Training Pipeline Completed Successfully!")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Model Location: {model_path}")
    print(f"Ready for deployment and predictions!")
    
    return best_model, comparison_df

if __name__ == "__main__":
    model, metrics = main()
