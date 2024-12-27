import pandas as pd
import numpy as np
import random

# Parameters
n_samples = 1000  # Number of data samples to generate

# Equipment types
equipment_types = ['MRI Scanner', 'Ventilator', 'CT Scanner', 'X-ray Machine', 'Blood Analyzer']

# Sensor and environment ranges
temperature_range = (18, 40)  # Extended to simulate extreme cases
humidity_range = (20, 80)
uptime_range = (100, 5000)
last_maintenance_range = (0, 730)  # 2 years max without maintenance

# Complex sensor ranges
vibration_range = (0, 6)
voltage_fluctuation_range = (0, 1.5)
pressure_range = (80, 130)
noise_factor = 0.05  # Adds random noise to sensor data

# Equipment failure thresholds
equipment_failure_conditions = {
    'MRI Scanner': {'temp_limit': 32, 'humidity_limit': 65, 'maintenance_limit': 300},
    'Ventilator': {'temp_limit': 30, 'humidity_limit': 60, 'maintenance_limit': 200},
    'CT Scanner': {'temp_limit': 28, 'humidity_limit': 55, 'maintenance_limit': 250},
    'X-ray Machine': {'temp_limit': 35, 'humidity_limit': 70, 'maintenance_limit': 365},
    'Blood Analyzer': {'temp_limit': 34, 'humidity_limit': 68, 'maintenance_limit': 350}
}

# Cascading failure effects (probability increments for sensor interactions)
sensor_cascade_effects = {
    'vibration': 0.15,
    'voltage_fluctuation': 0.2,
    'pressure': 0.1
}

# Generate random sensor data
data = []
for _ in range(n_samples):
    equipment_id = f"EQ-{random.randint(1000, 9999)}"
    equipment_type = random.choice(equipment_types)
    temp_avg = round(random.uniform(*temperature_range), 1)
    humidity_avg = round(random.uniform(*humidity_range), 1)
    last_maintenance_days = random.randint(*last_maintenance_range)
    uptime = random.randint(*uptime_range)

    # Simulate IoT sensor readings (with noise)
    sensor_1 = round(random.uniform(0, 1) + np.random.normal(0, noise_factor), 2)
    sensor_2 = round(random.uniform(0, 1) + np.random.normal(0, noise_factor), 2)
    vibration = round(random.uniform(*vibration_range) + np.random.normal(0, noise_factor), 2)
    voltage_fluctuation = round(random.uniform(*voltage_fluctuation_range) + np.random.normal(0, noise_factor), 2)
    pressure = round(random.uniform(*pressure_range) + np.random.normal(0, noise_factor), 1)

    # Base failure probability
    failure_probability = 0.05

    # Apply equipment-specific biases
    failure_conditions = equipment_failure_conditions[equipment_type]

    if temp_avg > failure_conditions['temp_limit']:
        failure_probability += 0.35
    if humidity_avg > failure_conditions['humidity_limit']:
        failure_probability += 0.25
    if last_maintenance_days > failure_conditions['maintenance_limit']:
        failure_probability += 0.3

    # Cascading sensor failures
    if vibration > 5:
        failure_probability += sensor_cascade_effects['vibration']
    if voltage_fluctuation > 1:
        failure_probability += sensor_cascade_effects['voltage_fluctuation']
    if pressure > 120:
        failure_probability += sensor_cascade_effects['pressure']

    # Environmental interaction
    if temp_avg > 36 and humidity_avg > 70:
        failure_probability += 0.2

    # Ensure failure probability is capped at 1.0
    failure_probability = min(round(failure_probability, 2), 1.0)

    # Equipment health assessment
    if failure_probability < 0.3:
        health = 'Good'
        maintenance_level = 'Low'
    elif failure_probability < 0.6:
        health = 'Moderate'
        maintenance_level = 'Medium'
    else:
        health = 'Critical'
        maintenance_level = 'High'

    # Failure reason and cost
    failure_reason = random.choice(['Overheating', 'Component Wear', 'Voltage Surge', 'Pressure Leak', 'Sensor Drift'])
    cost_implications = random.randint(1000, 15000)

    # Updated uptime after maintenance
    updated_uptime = uptime + random.randint(100, 1200)

    # Append to data
    data.append([
        equipment_id, equipment_type, temp_avg, humidity_avg,
        last_maintenance_days, uptime, sensor_1, sensor_2,
        vibration, voltage_fluctuation, pressure,
        health, failure_probability, failure_reason,
        maintenance_level, cost_implications, updated_uptime
    ])

# Create DataFrame
columns = [
    'equipment_id', 'equipment_type', 'temperature_avg', 'humidity_avg',
    'last_maintenance_days', 'uptime_hours', 'sensor_1', 'sensor_2',
    'vibration', 'voltage_fluctuation', 'pressure',
    'equipment_health', 'failure_probability', 'failure_reason',
    'maintenance_level', 'cost_implications', 'updated_uptime'
]
df = pd.DataFrame(data, columns=columns)
# Update failure types based on conditions
for i in range(len(df)):
    row = df.iloc[i]
    equipment_type = row['equipment_type']
    temp = row['temperature_avg']
    vibration = row['vibration']
    voltage = row['voltage_fluctuation']
    pressure = row['pressure']

    # Failure type bias
    if temp > 36 or vibration > 5:
        failure_type = 'Overheating'
    elif voltage > 1.2:
        failure_type = 'Voltage Surge'
    elif pressure > 120:
        failure_type = 'Pressure Leak'
    else:
        failure_type = random.choice(['Component Wear', 'Calibration Drift'])

    # Cost implications based on health status
    if row['equipment_health'] == 'Critical':
        cost = random.randint(8000, 15000)
    elif row['equipment_health'] == 'Moderate':
        cost = random.randint(4000, 8000)
    else:
        cost = random.randint(1000, 4000)

    # Reduce updated uptime for critical equipment
    if row['equipment_health'] == 'Critical':
        updated_uptime = row['uptime_hours'] + random.randint(100, 500)
    else:
        updated_uptime = row['uptime_hours'] + random.randint(600, 1200)

    # Update the row
    df.at[i, 'failure_reason'] = failure_type
    df.at[i, 'cost_implications'] = cost
    df.at[i, 'updated_uptime'] = updated_uptime

import argparse
parser = argparse.ArgumentParser(description='Generate synthetic healthcare equipment data.')
parser.add_argument('-op','--output_path', type=str, required=True, help='Output CSV file path')
args = parser.parse_args()

# Save DataFrame to CSV
df.to_csv(args.output_path, index=False)
print(f"Data saved to {args.output_path}")