import pandas as pd
import numpy as np
import joblib
import os

# Eco-friendly thresholds based on sustainable living standards
ECO_THRESHOLDS = {
    'transport_km_per_day': 15,      # Low-carbon transport target
    'electricity_kwh_per_month': 250, # Energy-efficient household
    'water_liters_per_day': 150,      # Water conservation target
    'waste_kg_per_week': 5,           # Minimal waste target
    'diet_emissions': {               # Daily CO2 emissions by diet
        'veg': 1.5,
        'mixed': 2.5,
        'non-veg': 3.3
    }
}

# Emission factors for carbon calculation
EMISSION_FACTORS = {
    'transport_co2_per_km': 0.12,     # kg CO2 per km (average car)
    'electricity_co2_per_kwh': 0.5,   # kg CO2 per kWh
    'water_co2_per_liter': 0.0003,    # kg CO2 per liter
    'waste_co2_per_kg': 0.5,          # kg CO2 per kg waste
}

class RecommendationEngine:
    """
    Generate personalized carbon footprint reduction recommendations.
    """
    
    def __init__(self, model_path='../models/carbon_model.pkl'):
        """Initialize the recommendation engine."""
        self.model = None
        self.model_path = model_path
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✓ Loaded prediction model from {model_path}")
    
    def calculate_carbon_footprint(self, transport_km, electricity_kwh, 
                                   water_liters, diet_type, waste_kg):
        """Calculate total carbon footprint."""
        # Transport emissions (monthly)
        transport_emissions = transport_km * EMISSION_FACTORS['transport_co2_per_km'] * 30
        
        # Electricity emissions (monthly)
        electricity_emissions = electricity_kwh * EMISSION_FACTORS['electricity_co2_per_kwh']
        
        # Water emissions (monthly)
        water_emissions = water_liters * EMISSION_FACTORS['water_co2_per_liter'] * 30
        
        # Diet emissions (monthly)
        diet_emissions_daily = ECO_THRESHOLDS['diet_emissions'].get(diet_type, 2.5)
        diet_emissions = diet_emissions_daily * 30
        
        # Waste emissions (monthly)
        waste_emissions = waste_kg * EMISSION_FACTORS['waste_co2_per_kg'] * 4
        
        total = (transport_emissions + electricity_emissions + 
                water_emissions + diet_emissions + waste_emissions)
        
        return {
            'total': total,
            'transport': transport_emissions,
            'electricity': electricity_emissions,
            'water': water_emissions,
            'diet': diet_emissions,
            'waste': waste_emissions
        }
    
    def get_transport_recommendations(self, current_km):
        """Generate transport-related recommendations."""
        recommendations = []
        threshold = ECO_THRESHOLDS['transport_km_per_day']
        
        if current_km > threshold:
            excess_km = current_km - threshold
            potential_savings = excess_km * EMISSION_FACTORS['transport_co2_per_km'] * 30
            
            # Main recommendation
            recommendations.append({
                'category': 'Transport',
                'priority': 'High',
                'current_value': f"{current_km:.1f} km/day",
                'target_value': f"{threshold} km/day",
                'suggestion': f"Reduce car usage by {excess_km:.1f} km/day to save {potential_savings:.1f} kg CO₂/month",
                'potential_savings_kg_co2': round(potential_savings, 2),
                'impact': 'High' if excess_km > 20 else 'Medium'
            })
            
            # Specific alternatives
            if excess_km >= 5:
                savings_5km = 5 * EMISSION_FACTORS['transport_co2_per_km'] * 30
                recommendations.append({
                    'category': 'Transport',
                    'priority': 'Medium',
                    'suggestion': f"Use public transport for 5 km/day to save {savings_5km:.1f} kg CO₂/month",
                    'potential_savings_kg_co2': round(savings_5km, 2),
                    'impact': 'Medium'
                })
            
            if excess_km >= 3:
                savings_cycling = 3 * EMISSION_FACTORS['transport_co2_per_km'] * 30
                recommendations.append({
                    'category': 'Transport',
                    'priority': 'Medium',
                    'suggestion': f"Cycle or walk for short trips (3 km/day) to save {savings_cycling:.1f} kg CO₂/month",
                    'potential_savings_kg_co2': round(savings_cycling, 2),
                    'impact': 'Medium'
                })
            
            recommendations.append({
                'category': 'Transport',
                'priority': 'Medium',
                'suggestion': "Consider carpooling to reduce individual emissions by 50%",
                'potential_savings_kg_co2': round(potential_savings * 0.5, 2),
                'impact': 'High'
            })
        else:
            recommendations.append({
                'category': 'Transport',
                'priority': 'Low',
                'suggestion': f"✓ Great job! Your transport usage ({current_km:.1f} km/day) is below the eco-friendly threshold",
                'potential_savings_kg_co2': 0,
                'impact': 'Positive'
            })
        
        return recommendations
    
    def get_electricity_recommendations(self, current_kwh):
        """Generate electricity-related recommendations."""
        recommendations = []
        threshold = ECO_THRESHOLDS['electricity_kwh_per_month']
        
        if current_kwh > threshold:
            excess_kwh = current_kwh - threshold
            potential_savings = excess_kwh * EMISSION_FACTORS['electricity_co2_per_kwh']
            
            recommendations.append({
                'category': 'Electricity',
                'priority': 'High',
                'current_value': f"{current_kwh:.1f} kWh/month",
                'target_value': f"{threshold} kWh/month",
                'suggestion': f"Reduce electricity consumption by {excess_kwh:.1f} kWh/month to save {potential_savings:.1f} kg CO₂/month",
                'potential_savings_kg_co2': round(potential_savings, 2),
                'impact': 'High' if excess_kwh > 150 else 'Medium'
            })
            
            # Specific actions
            if excess_kwh >= 50:
                savings_led = 50 * EMISSION_FACTORS['electricity_co2_per_kwh']
                recommendations.append({
                    'category': 'Electricity',
                    'priority': 'High',
                    'suggestion': f"Switch to LED bulbs and energy-efficient appliances (save ~50 kWh = {savings_led:.1f} kg CO₂/month)",
                    'potential_savings_kg_co2': round(savings_led, 2),
                    'impact': 'Medium'
                })
            
            if excess_kwh >= 30:
                savings_ac = 30 * EMISSION_FACTORS['electricity_co2_per_kwh']
                recommendations.append({
                    'category': 'Electricity',
                    'priority': 'Medium',
                    'suggestion': f"Optimize AC/heating usage (set thermostat 2°C higher/lower = {savings_ac:.1f} kg CO₂/month saved)",
                    'potential_savings_kg_co2': round(savings_ac, 2),
                    'impact': 'Medium'
                })
            
            recommendations.append({
                'category': 'Electricity',
                'priority': 'Medium',
                'suggestion': "Unplug electronics when not in use to reduce phantom power drain (5-10% savings)",
                'potential_savings_kg_co2': round(current_kwh * 0.075 * EMISSION_FACTORS['electricity_co2_per_kwh'], 2),
                'impact': 'Low'
            })
        else:
            recommendations.append({
                'category': 'Electricity',
                'priority': 'Low',
                'suggestion': f"✓ Excellent! Your electricity usage ({current_kwh:.1f} kWh/month) is eco-friendly",
                'potential_savings_kg_co2': 0,
                'impact': 'Positive'
            })
        
        return recommendations
    
    def get_water_recommendations(self, current_liters):
        """Generate water conservation recommendations."""
        recommendations = []
        threshold = ECO_THRESHOLDS['water_liters_per_day']
        
        if current_liters > threshold:
            excess_liters = current_liters - threshold
            potential_savings = excess_liters * EMISSION_FACTORS['water_co2_per_liter'] * 30
            
            recommendations.append({
                'category': 'Water',
                'priority': 'Medium',
                'current_value': f"{current_liters:.1f} L/day",
                'target_value': f"{threshold} L/day",
                'suggestion': f"Reduce water usage by {excess_liters:.1f} L/day to save {potential_savings:.2f} kg CO₂/month",
                'potential_savings_kg_co2': round(potential_savings, 2),
                'impact': 'Low' if excess_liters < 50 else 'Medium'
            })
            
            if excess_liters >= 20:
                savings_shower = 20 * EMISSION_FACTORS['water_co2_per_liter'] * 30
                recommendations.append({
                    'category': 'Water',
                    'priority': 'Medium',
                    'suggestion': f"Take shorter showers (reduce by 5 minutes = 20 L/day saved = {savings_shower:.2f} kg CO₂/month)",
                    'potential_savings_kg_co2': round(savings_shower, 2),
                    'impact': 'Low'
                })
            
            recommendations.append({
                'category': 'Water',
                'priority': 'Low',
                'suggestion': "Install water-efficient fixtures and fix leaks to save 10-15% water",
                'potential_savings_kg_co2': round(current_liters * 0.125 * EMISSION_FACTORS['water_co2_per_liter'] * 30, 2),
                'impact': 'Low'
            })
        else:
            recommendations.append({
                'category': 'Water',
                'priority': 'Low',
                'suggestion': f"✓ Good work! Your water usage ({current_liters:.1f} L/day) is sustainable",
                'potential_savings_kg_co2': 0,
                'impact': 'Positive'
            })
        
        return recommendations
    
    def get_diet_recommendations(self, diet_type):
        """Generate diet-related recommendations."""
        recommendations = []
        diet_emissions = ECO_THRESHOLDS['diet_emissions']
        
        if diet_type == 'non-veg':
            savings = (diet_emissions['non-veg'] - diet_emissions['mixed']) * 30
            recommendations.append({
                'category': 'Diet',
                'priority': 'High',
                'current_value': 'Non-vegetarian',
                'suggestion': f"Reduce meat consumption to 3-4 days/week (mixed diet) to save {savings:.1f} kg CO₂/month",
                'potential_savings_kg_co2': round(savings, 2),
                'impact': 'High'
            })
            
            savings_veg = (diet_emissions['non-veg'] - diet_emissions['veg']) * 30
            recommendations.append({
                'category': 'Diet',
                'priority': 'Medium',
                'suggestion': f"Switch to vegetarian diet to save {savings_veg:.1f} kg CO₂/month",
                'potential_savings_kg_co2': round(savings_veg, 2),
                'impact': 'High'
            })
            
            recommendations.append({
                'category': 'Diet',
                'priority': 'Medium',
                'suggestion': "Choose chicken or fish instead of red meat (50% lower emissions)",
                'potential_savings_kg_co2': round(savings * 0.5, 2),
                'impact': 'Medium'
            })
            
        elif diet_type == 'mixed':
            savings = (diet_emissions['mixed'] - diet_emissions['veg']) * 30
            recommendations.append({
                'category': 'Diet',
                'priority': 'Medium',
                'current_value': 'Mixed diet',
                'suggestion': f"Try 'Meatless Mondays' or reduce meat intake to save up to {savings:.1f} kg CO₂/month",
                'potential_savings_kg_co2': round(savings, 2),
                'impact': 'Medium'
            })
            
            recommendations.append({
                'category': 'Diet',
                'priority': 'Low',
                'suggestion': "Choose locally sourced and seasonal produce to reduce food transport emissions",
                'potential_savings_kg_co2': round(savings * 0.2, 2),
                'impact': 'Low'
            })
            
        else:  # vegetarian
            recommendations.append({
                'category': 'Diet',
                'priority': 'Low',
                'suggestion': f"✓ Amazing! Your vegetarian diet is already low-carbon",
                'potential_savings_kg_co2': 0,
                'impact': 'Positive'
            })
            
            recommendations.append({
                'category': 'Diet',
                'priority': 'Low',
                'suggestion': "Continue with plant-based choices and consider reducing dairy for even lower impact",
                'potential_savings_kg_co2': 5,
                'impact': 'Low'
            })
        
        return recommendations
    
    def get_waste_recommendations(self, current_waste):
        """Generate waste reduction recommendations."""
        recommendations = []
        threshold = ECO_THRESHOLDS['waste_kg_per_week']
        
        if current_waste > threshold:
            excess_waste = current_waste - threshold
            potential_savings = excess_waste * EMISSION_FACTORS['waste_co2_per_kg'] * 4
            
            recommendations.append({
                'category': 'Waste',
                'priority': 'High',
                'current_value': f"{current_waste:.1f} kg/week",
                'target_value': f"{threshold} kg/week",
                'suggestion': f"Reduce waste by {excess_waste:.1f} kg/week to save {potential_savings:.1f} kg CO₂/month",
                'potential_savings_kg_co2': round(potential_savings, 2),
                'impact': 'Medium' if excess_waste > 5 else 'Low'
            })
            
            if excess_waste >= 3:
                savings_recycle = 3 * EMISSION_FACTORS['waste_co2_per_kg'] * 4
                recommendations.append({
                    'category': 'Waste',
                    'priority': 'High',
                    'suggestion': f"Start composting organic waste (reduce 3 kg/week = {savings_recycle:.1f} kg CO₂/month saved)",
                    'potential_savings_kg_co2': round(savings_recycle, 2),
                    'impact': 'Medium'
                })
            
            recommendations.append({
                'category': 'Waste',
                'priority': 'Medium',
                'suggestion': "Use reusable bags, bottles, and containers to reduce packaging waste by 20-30%",
                'potential_savings_kg_co2': round(current_waste * 0.25 * EMISSION_FACTORS['waste_co2_per_kg'] * 4, 2),
                'impact': 'Medium'
            })
            
            recommendations.append({
                'category': 'Waste',
                'priority': 'Medium',
                'suggestion': "Practice proper recycling to divert 40-50% of waste from landfills",
                'potential_savings_kg_co2': round(current_waste * 0.45 * EMISSION_FACTORS['waste_co2_per_kg'] * 4, 2),
                'impact': 'Medium'
            })
        else:
            recommendations.append({
                'category': 'Waste',
                'priority': 'Low',
                'suggestion': f"✓ Perfect! Your waste generation ({current_waste:.1f} kg/week) is minimal",
                'potential_savings_kg_co2': 0,
                'impact': 'Positive'
            })
        
        return recommendations
    
    def generate_recommendations(self, transport_km, electricity_kwh, 
                                water_liters, diet_type, waste_kg):
        """
        Generate comprehensive personalized recommendations.
        
        Parameters:
        -----------
        transport_km : float
            Daily transport distance in km
        electricity_kwh : float
            Monthly electricity consumption in kWh
        water_liters : float
            Daily water usage in liters
        diet_type : str
            Diet type: 'veg', 'mixed', or 'non-veg'
        waste_kg : float
            Weekly waste generation in kg
        
        Returns:
        --------
        dict : Contains current footprint, recommendations list, and summary
        """
        
        # Calculate current carbon footprint
        footprint = self.calculate_carbon_footprint(
            transport_km, electricity_kwh, water_liters, diet_type, waste_kg
        )
        
        # Generate recommendations by category
        all_recommendations = []
        all_recommendations.extend(self.get_transport_recommendations(transport_km))
        all_recommendations.extend(self.get_electricity_recommendations(electricity_kwh))
        all_recommendations.extend(self.get_water_recommendations(water_liters))
        all_recommendations.extend(self.get_diet_recommendations(diet_type))
        all_recommendations.extend(self.get_waste_recommendations(waste_kg))
        
        # Calculate total potential savings
        total_savings = sum(r.get('potential_savings_kg_co2', 0) 
                          for r in all_recommendations)
        
        # Sort by priority and impact
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        all_recommendations.sort(
            key=lambda x: (priority_order.get(x.get('priority', 'Low'), 3),
                          -x.get('potential_savings_kg_co2', 0))
        )
        
        # Create summary
        summary = {
            'current_footprint_kg_co2': round(footprint['total'], 2),
            'footprint_breakdown': {k: round(v, 2) for k, v in footprint.items()},
            'total_potential_savings_kg_co2': round(total_savings, 2),
            'potential_footprint_kg_co2': round(footprint['total'] - total_savings, 2),
            'reduction_percentage': round((total_savings / footprint['total']) * 100, 2) if footprint['total'] > 0 else 0,
            'top_contributor': max(footprint, key=lambda k: footprint[k] if k != 'total' else 0),
            'eco_score': self._calculate_eco_score(transport_km, electricity_kwh, 
                                                   water_liters, diet_type, waste_kg)
        }
        
        return {
            'recommendations': all_recommendations,
            'summary': summary
        }
    
    def _calculate_eco_score(self, transport_km, electricity_kwh, 
                            water_liters, diet_type, waste_kg):
        """Calculate an eco-friendliness score (0-100)."""
        scores = []
        
        # Transport score
        transport_score = max(0, 100 - (transport_km / ECO_THRESHOLDS['transport_km_per_day']) * 100 + 100)
        scores.append(min(100, transport_score))
        
        # Electricity score
        elec_score = max(0, 100 - (electricity_kwh / ECO_THRESHOLDS['electricity_kwh_per_month']) * 100 + 100)
        scores.append(min(100, elec_score))
        
        # Water score
        water_score = max(0, 100 - (water_liters / ECO_THRESHOLDS['water_liters_per_day']) * 100 + 100)
        scores.append(min(100, water_score))
        
        # Diet score
        diet_scores = {'veg': 100, 'mixed': 65, 'non-veg': 35}
        scores.append(diet_scores.get(diet_type, 50))
        
        # Waste score
        waste_score = max(0, 100 - (waste_kg / ECO_THRESHOLDS['waste_kg_per_week']) * 100 + 100)
        scores.append(min(100, waste_score))
        
        return round(sum(scores) / len(scores), 1)


def demo_recommendation_engine():
    """Demonstrate the recommendation engine with sample user data."""
    print("="*70)
    print("Green Habit Recommendation Engine - Demo")
    print("="*70)
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Sample user profiles
    users = [
        {
            'name': 'User 1 (High Carbon)',
            'transport_km': 45,
            'electricity_kwh': 450,
            'water_liters': 250,
            'diet_type': 'non-veg',
            'waste_kg': 12
        },
        {
            'name': 'User 2 (Medium Carbon)',
            'transport_km': 25,
            'electricity_kwh': 320,
            'water_liters': 180,
            'diet_type': 'mixed',
            'waste_kg': 7
        },
        {
            'name': 'User 3 (Low Carbon)',
            'transport_km': 10,
            'electricity_kwh': 200,
            'water_liters': 130,
            'diet_type': 'veg',
            'waste_kg': 4
        }
    ]
    
    for user in users:
        print(f"\n{'='*70}")
        print(f"{user['name']}")
        print(f"{'='*70}")
        print(f"Input Values:")
        print(f"  - Transport: {user['transport_km']} km/day")
        print(f"  - Electricity: {user['electricity_kwh']} kWh/month")
        print(f"  - Water: {user['water_liters']} L/day")
        print(f"  - Diet: {user['diet_type']}")
        print(f"  - Waste: {user['waste_kg']} kg/week")
        
        # Generate recommendations
        result = engine.generate_recommendations(
            user['transport_km'], user['electricity_kwh'],
            user['water_liters'], user['diet_type'], user['waste_kg']
        )
        
        # Display summary
        print(f"\n{'─'*70}")
        print("Carbon Footprint Analysis:")
        print(f"{'─'*70}")
        print(f"Current Footprint: {result['summary']['current_footprint_kg_co2']} kg CO₂/month")
        print(f"Potential Savings: {result['summary']['total_potential_savings_kg_co2']} kg CO₂/month")
        print(f"Potential Footprint: {result['summary']['potential_footprint_kg_co2']} kg CO₂/month")
        print(f"Reduction Potential: {result['summary']['reduction_percentage']}%")
        print(f"Eco Score: {result['summary']['eco_score']}/100")
        print(f"Top Contributor: {result['summary']['top_contributor'].replace('_', ' ').title()}")
        
        # Display top recommendations
        print(f"\n{'─'*70}")
        print("Top 5 Personalized Recommendations:")
        print(f"{'─'*70}")
        for i, rec in enumerate(result['recommendations'][:5], 1):
            priority_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(rec['priority'], '⚪')
            print(f"\n{i}. [{priority_icon} {rec['priority']}] {rec['category']}")
            print(f"   {rec['suggestion']}")
            if rec.get('potential_savings_kg_co2', 0) > 0:
                print(f"   💡 Savings: {rec['potential_savings_kg_co2']} kg CO₂/month")
    
    print(f"\n{'='*70}")
    print("Demo completed successfully!")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo_recommendation_engine()
