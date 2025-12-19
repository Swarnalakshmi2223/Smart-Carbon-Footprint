import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_timeseries_data(data_path='../data/carbon_footprint.csv'):
    """Load data and convert to time-series format."""
    print("Loading carbon footprint data...")
    df = pd.read_csv(data_path)
    
    # Create a synthetic time series by sorting and adding dates
    # Simulate monthly data points starting from Jan 2023
    start_date = pd.date_range(start='2023-01-01', periods=len(df), freq='M')
    df['date'] = start_date
    
    # Sort by carbon footprint to create a realistic trend
    df = df.sort_values('carbon_footprint_kg_co2').reset_index(drop=True)
    
    # Add some temporal patterns (seasonal variations)
    df['month'] = pd.to_datetime(df['date']).dt.month
    seasonal_factor = np.sin(2 * np.pi * df['month'] / 12) * 20  # +/- 20 kg variation
    df['carbon_footprint_kg_co2'] = df['carbon_footprint_kg_co2'] + seasonal_factor
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"  - Time series length: {len(df)} months")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def create_sequences(data, sequence_length=12):
    """Create sequences for LSTM model."""
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    
    return np.array(X), np.array(y)

def prepare_lstm_data(df, sequence_length=12, train_split=0.8):
    """Prepare data for LSTM training."""
    print(f"\nPreparing LSTM sequences (sequence length: {sequence_length})...")
    
    # Extract carbon footprint values
    carbon_data = df['carbon_footprint_kg_co2'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    carbon_scaled = scaler.fit_transform(carbon_data)
    
    # Create sequences
    X, y = create_sequences(carbon_scaled, sequence_length)
    
    # Split into train and test
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"  - Training sequences: {len(X_train)}")
    print(f"  - Test sequences: {len(X_test)}")
    print(f"  - Sequence shape: {X_train.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, carbon_scaled

def build_lstm_model(sequence_length, learning_rate=0.001):
    """Build LSTM model architecture."""
    print("\nBuilding LSTM model...")
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(sequence_length, 1)),
        Dropout(0.2),
        
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.2),
        
        LSTM(16, activation='relu'),
        Dropout(0.2),
        
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    print("  ✓ Model architecture created")
    print(f"\n{model.summary()}")
    
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Train the LSTM model."""
    print("\nTraining LSTM model...")
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n  ✓ Training completed")
    
    return history

def predict_future(model, last_sequence, scaler, months_ahead=6):
    """Predict carbon footprint for the next N months."""
    print(f"\nGenerating predictions for next {months_ahead} months...")
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for i in range(months_ahead):
        # Predict next value
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence (remove first, add prediction)
        current_sequence = np.append(current_sequence[1:], pred[0, 0])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions)
    
    print(f"  ✓ Predictions generated")
    
    return predictions_original.flatten()

def plot_training_history(history, output_dir='../reports'):
    """Plot training and validation loss."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (MSE)', fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model MAE During Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('MAE', fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_training_history.png'), dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Training history saved to: {output_dir}/lstm_training_history.png")
    plt.show()

def plot_predictions(df, predictions, sequence_length=12, output_dir='../reports'):
    """Plot historical data and future predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get last date from data
    last_date = pd.to_datetime(df['date'].iloc[-1])
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=len(predictions),
        freq='M'
    )
    
    # Plot
    plt.figure(figsize=(16, 7))
    
    # Historical data (last 36 months for clarity)
    historical_window = min(36, len(df))
    plt.plot(df['date'].iloc[-historical_window:], 
             df['carbon_footprint_kg_co2'].iloc[-historical_window:],
             label='Historical Data', linewidth=2, color='steelblue', marker='o', markersize=4)
    
    # Future predictions
    plt.plot(future_dates, predictions,
             label='Future Predictions', linewidth=2.5, color='red', 
             marker='s', markersize=6, linestyle='--')
    
    # Add vertical line to separate historical and future
    plt.axvline(x=last_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(last_date, plt.ylim()[1]*0.95, 'Present', 
             ha='right', va='top', fontsize=11, fontweight='bold')
    
    plt.title('Carbon Footprint: Historical Data & 6-Month Forecast', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Carbon Footprint (kg CO₂/month)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_future_predictions.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Predictions plot saved to: {output_dir}/lstm_future_predictions.png")
    plt.show()

def plot_comparison(df, model, X_test, y_test, scaler, sequence_length, output_dir='../reports'):
    """Plot model predictions vs actual values on test set."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred)
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(y_test_original, label='Actual', linewidth=2, marker='o', markersize=4)
    plt.plot(y_pred_original, label='Predicted', linewidth=2, marker='s', 
             markersize=4, alpha=0.7)
    
    plt.title('LSTM Model: Actual vs Predicted (Test Set)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Carbon Footprint (kg CO₂)', fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    # Calculate and display metrics
    mae = np.mean(np.abs(y_test_original - y_pred_original))
    rmse = np.sqrt(np.mean((y_test_original - y_pred_original)**2))
    
    plt.text(0.02, 0.98, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_test_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Comparison plot saved to: {output_dir}/lstm_test_comparison.png")
    plt.show()
    
    return mae, rmse

def save_predictions(predictions, output_dir='../reports'):
    """Save future predictions to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create future dates
    future_dates = pd.date_range(
        start='2026-01-01',
        periods=len(predictions),
        freq='M'
    )
    
    predictions_df = pd.DataFrame({
        'date': future_dates,
        'predicted_carbon_footprint_kg_co2': predictions
    })
    
    output_path = os.path.join(output_dir, 'future_predictions.csv')
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\n  ✓ Predictions saved to: {output_path}")
    print("\n  Future Predictions:")
    print(predictions_df.to_string(index=False))

def main():
    """Main LSTM prediction pipeline."""
    print("="*70)
    print("Carbon Footprint Time Series Forecasting - LSTM Model")
    print("="*70)
    
    # Configuration
    SEQUENCE_LENGTH = 12  # Use 12 months to predict next month
    EPOCHS = 100
    BATCH_SIZE = 32
    FUTURE_MONTHS = 6
    
    # Load and prepare data
    df = load_and_prepare_timeseries_data()
    
    # Prepare LSTM data
    X_train, X_test, y_train, y_test, scaler, carbon_scaled = prepare_lstm_data(
        df, sequence_length=SEQUENCE_LENGTH
    )
    
    # Build model
    model = build_lstm_model(sequence_length=SEQUENCE_LENGTH)
    
    # Train model
    history = train_lstm_model(
        model, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_mae, test_rmse = plot_comparison(
        df, model, X_test, y_test, scaler, SEQUENCE_LENGTH
    )
    
    print(f"\n{'='*70}")
    print("Test Set Performance:")
    print(f"  - MAE:  {test_mae:.4f} kg CO₂")
    print(f"  - RMSE: {test_rmse:.4f} kg CO₂")
    print(f"{'='*70}")
    
    # Predict future
    last_sequence = carbon_scaled[-SEQUENCE_LENGTH:]
    future_predictions = predict_future(
        model, last_sequence, scaler, months_ahead=FUTURE_MONTHS
    )
    
    # Plot predictions
    plot_predictions(df, future_predictions, SEQUENCE_LENGTH)
    
    # Save predictions
    save_predictions(future_predictions)
    
    # Save model
    model_path = '../models/lstm_carbon_model.h5'
    os.makedirs('../models', exist_ok=True)
    model.save(model_path)
    print(f"\n  ✓ LSTM model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("LSTM Forecasting Pipeline Completed Successfully!")
    print("="*70)
    print(f"\nNext 6 months average prediction: {np.mean(future_predictions):.2f} kg CO₂")
    print(f"Predicted trend: {'Increasing' if future_predictions[-1] > future_predictions[0] else 'Decreasing'}")
    
    return model, future_predictions, history

if __name__ == "__main__":
    model, predictions, history = main()
