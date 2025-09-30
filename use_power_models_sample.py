#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample code for Server and HVAC power prediction

"""

from pycaret.regression import load_model  
import pandas as pd  

#%% Example: server models for Edge side

# Load the pre-trained Server model  
model_server = load_model('server_model_supermicro')  # Load the edge supermicor server model for power prediction  
# model_server = load_model('server_model_lenovo')      # Load the edge lenovo model for power prediction  

# Create sample input data for making the prediction  
data = {  
    "cpu": [20],            #  CPU ratio (0.1 ~ 100)  
    "amb_temp": [21],       #  Server ambient temperature (C)  
    "dp": [2.5]             #  Server statis pressure difference(Pa)  
}  

# Convert the input data into a DataFrame  
input_df_server_pw = pd.DataFrame(data)  

# Predict server power consumption using the loaded model  
PW_Server = model_server.predict(input_df_server_pw)[0]  # Predicted server power (Watt)  

# Print the predicted Server power  
print(f"Predicted Server Power at edge site: {PW_Server:.2f} W")  


#%% Example: server models for Cloud side

model_server = load_model('server_model_cloud_dell')    # Load the cloud dell server model for power prediction  
# model_server = load_model('server_model_cloud_lenovo')  # Load the cloud lenovo server model for power prediction  

# Create sample input data for making the prediction  
data = {  
    "mean_cpu": [20],            #  CPU ratio (0.1 ~ 100)  
    "amb_temp": [21],            #  Server ambient temperature (C)  
    "mean_dp": [2.5]             #  Server statis pressure difference(Pa)  
}  

# Convert the input data into a DataFrame  
input_df_server_pw = pd.DataFrame(data)  

# Predict server power consumption using the loaded model  
PW_Server = model_server.predict(input_df_server_pw)[0]  # Predicted server power (Watt)  

# Print the predicted Server power  
print(f"Predicted Server Power at cloud site: {PW_Server:.2f} W")  

#%% Example: HVAC models for edge side

# Load the pre-trained HVAC model  
model_HVAC = load_model('HVAC_model')  # Load the HVAC model for power prediction  

# Create sample input data for making the prediction  
data = {  
    "temp_chiller_water": [20],       # Chiller water temperature (°C)  
    "fan_speed_indoor_1": [1300],     # Indoor HVAC fan speed (RPM)  
    "humid_outdoor": [61.0],          # Outdoor humidity (%)  
    "temp_outdoor": [26.15],          # Outdoor temperature (°C)  
    "power_server": [16.6],           # Total server power consumption (kW)  
    "temp_supply_air": [25],          # HVAC air supply temperature (°C)  
    "dp_hvac_1": [72.0]               # HVAC static pressure difference (Pa)  
}  

# Convert the input data into a DataFrame  
input_df_hvac_pw = pd.DataFrame(data)  

# Predict HVAC power consumption using the loaded model  
PW_hvac = model_HVAC.predict(input_df_hvac_pw)[0]  # Predicted HVAC power (kW)  

# Print the predicted HVAC power  
print(f"Predicted HVAC Power: {PW_hvac:.2f} kW")  