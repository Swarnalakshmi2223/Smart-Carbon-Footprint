import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 samples
n_samples = 1000

# Generate features with realistic distributions

# Transport: 0-100 km per day (average ~20-30 km)
transport_km_per_day = np.random.gamma(shape=2, scale=15, size=n_samples)
transport_km_per_day = np.clip(transport_km_per_day, 0, 150)

# Electricity: 100-800 kWh per month (average household ~300-400 kWh)
electricity_kwh_per_month = np.random.normal(loc=350, scale=120, size=n_samples)
electricity_kwh_per_month = np.clip(electricity_kwh_per_month, 50, 1000)

# Water: 50-500 liters per day (average ~150-200 liters per person)
water_liters_per_day = np.random.normal(loc=180, scale=60, size=n_samples)
water_liters_per_day = np.clip(water_liters_per_day, 30, 500)

# Diet type: veg, mixed, non-veg (with realistic distribution)
diet_types = ['veg', 'mixed', 'non-veg']
diet_probabilities = [0.25, 0.55, 0.20]  # 25% veg, 55% mixed, 20% non-veg
diet_type = np.random.choice(diet_types, size=n_samples, p=diet_probabilities)

# Waste: 2-20 kg per week (average ~5-8 kg)
waste_kg_per_week = np.random.gamma(shape=3, scale=2.5, size=n_samples)
waste_kg_per_week = np.clip(waste_kg_per_week, 1, 25)

# Calculate carbon footprint (kg CO2 per month)
# Emission factors (approximate):
# - Transport: 0.12 kg CO2 per km (car average)
# - Electricity: 0.5 kg CO2 per kWh (grid average)
# - Water: 0.0003 kg CO2 per liter (treatment & heating)
# - Diet: veg=1.5, mixed=2.5, non-veg=3.3 kg CO2 per day
# - Waste: 0.5 kg CO2 per kg (landfill emissions)

carbon_footprint_kg_co2 = np.zeros(n_samples)

for i in range(n_samples):
    # Transport emissions (monthly)
    transport_emissions = transport_km_per_day[i] * 0.12 * 30
    
    # Electricity emissions (monthly)
    electricity_emissions = electricity_kwh_per_month[i] * 0.5
    
    # Water emissions (monthly)
    water_emissions = water_liters_per_day[i] * 0.0003 * 30
    
    # Diet emissions (monthly)
    if diet_type[i] == 'veg':
        diet_emissions = 1.5 * 30
    elif diet_type[i] == 'mixed':
        diet_emissions = 2.5 * 30
    else:  # non-veg
        diet_emissions = 3.3 * 30
    
    # Waste emissions (monthly)
    waste_emissions = waste_kg_per_week[i] * 0.5 * 4
    
    # Total carbon footprint
    carbon_footprint_kg_co2[i] = (
        transport_emissions + 
        electricity_emissions + 
        water_emissions + 
        diet_emissions + 
        waste_emissions
    )
    
    # Add some realistic noise (±5%)
    noise = np.random.normal(1.0, 0.05)
    carbon_footprint_kg_co2[i] *= noise

# Create DataFrame
df = pd.DataFrame({
    'transport_km_per_day': np.round(transport_km_per_day, 2),
    'electricity_kwh_per_month': np.round(electricity_kwh_per_month, 2),
    'water_liters_per_day': np.round(water_liters_per_day, 2),
    'diet_type': diet_type,
    'waste_kg_per_week': np.round(waste_kg_per_week, 2),
    'carbon_footprint_kg_co2': np.round(carbon_footprint_kg_co2, 2)
})

# Save to CSV
df.to_csv('carbon_footprint.csv', index=False)

print(f"Dataset generated successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())
print(f"\nDiet type distribution:")
print(df['diet_type'].value_counts())
print(f"\nDataset saved to: carbon_footprint.csv")
