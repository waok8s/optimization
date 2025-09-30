"""

key points:
 
@author: ying-fenghsu
"""

import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import pickle
from pycaret.regression import load_model
from joblib import Parallel, delayed


init_temp_chiller_water = 16 # temp_supply_air   , adjust as need
init_fan_speed = 1400 # fan speed, adjust as need


# Set the logging level to WARNING (this will disable Optuna's optimization progress display in screen)
optuna.logging.set_verbosity(optuna.logging.WARNING)

#% Load hvac models 
model_HVAC = load_model('../hotspot_model_HVAC_power_0731')  #load_model('model_HVAC_ExtraTreeRegressor_0206') 
model_HVAC_AirTemp_load = load_model('../hotspot_model_HVAC_AirTemp_0731') #load_model('model_HVAC_AirTemp_ExtraTreeRegressor_0206')

# load server models
model_eLenovo_load = load_model('../hotspot_model_lenovo')
model_esupermicro_load = load_model('../hotspot_model_supermicro')

# server backplane temp model informaion
save_path = '../backplane_temp_models/'
file_name = 'summary_models.csv'
file_path = os.path.join(save_path, file_name)
model_backplane_temp_summary = pd.read_csv(file_path)
Edge_site_neighbor = pd.read_csv('../Edge_site_neighbor.csv')
backplane_temp_list = pd.merge(Edge_site_neighbor[['host_name', 'Machine_type', 'host_name_D', 'host_name_U']], model_backplane_temp_summary, on='host_name', how='left')
backplane_temp_list = backplane_temp_list.drop(columns=['machine_type'])
backplane_temp_list.rename(columns={'Machine_type': 'machine_type'}, inplace=True)

# preload backplane models
# Create a dictionary to store the models
model_dict = {}
for model_filename in model_backplane_temp_summary['model_filename'].unique():
    model_path = os.path.join(save_path, model_filename)
    model_dict[model_filename] = load_model(model_path, verbose=False)


#load  simulate data
df_HVAC = pd.read_csv('../average_data_180_Y2024.csv')
df_workload_kagoya = pd.read_csv('../kagoyaDC_workload.csv')
df_workload_kagoya['load'] = df_workload_kagoya['load'] * 100


# server dp reference table
dp_server_ref_table = pd.read_csv('../hotspot_ref_DP_forPython.csv')
sever_list = dp_server_ref_table[['host_name', 'machine_type']].drop_duplicates()
dp_hvac_ref_table = pd.read_csv('../hotspot_ref_DP_hvac_table.csv')

# Global variables to store history
pred_PW_all_Server_hist = []
pred_PW_all_Lenovo_hist = []
pred_PW_all_Supermicro_hist = []
pred_TempSupplyAir_hist = []
pred_PW_HVAC_hist = []
pred_PW_HVAC_ref_hist = []
Control_FanSpeed_hist = []
Control_ChillerWaterTemp_hist = []
backplane_temp_lenovo_hist =[]
backplane_temp_supermicro_hist = []
all_trials_df_hist_list =[]
#%% function block
def find_closest_speed(df, input_speed):
    # Create an empty list to store the results
    results = []

    # Group by 'host_name'
    grouped = df.groupby('host_name')

    # Iterate over each group
    for host, group in grouped:
        # Calculate the absolute difference between 'input_speed' and 'span_speed_indoor_1'
        group['difference'] = abs(group['fan_speed_indoor_1'] - input_speed)
        
        # Find the row with the minimum difference
        min_diff_row = group.loc[group['difference'].idxmin()]

        # Append the result to the list
        results.append({
            'host_name': host,
            'input_speed': input_speed,
            'closest_to_span_speed_indoor_1': min_diff_row['fan_speed_indoor_1'],
            'mean_dp': min_diff_row['mean_dp']
        })

    # Convert the results list to a DataFrame
    result_df = pd.DataFrame(results)
    return result_df

def find_nearest_dp_hvac(input_value):  
    # Calculate the absolute differences  
    dp_hvac_ref_table['difference'] = (dp_hvac_ref_table['fan_speed_indoor_1_rounded'] - input_value).abs()  
    
    # Find the index of the minimum difference  
    nearest_index = dp_hvac_ref_table['difference'].idxmin()  
    
    # Return the corresponding dp_hvac_1 value  
    return dp_hvac_ref_table.loc[nearest_index, 'dp_hvac_1']  

def compute_backplane_temp(input_server_status):
    
    #generate input table
    # 6 inputs ['lower_cpu', 'upper_cpu', 'middle_cpu', 'lower_amb_temp', 'upper_amb_temp', 'middle_amb_temp',  'middle_back_temp']
    A = pd.merge(backplane_temp_list, input_server_status[['host_name', 'machine_type', 'amb_temp', 'cpu']], on='host_name', how='left')
        
    # Second merge
    A1 = pd.merge(A, input_server_status[['host_name', 'cpu']], left_on='host_name_D', right_on='host_name', how='left')
    # 3rd merge
    A2 = pd.merge(A1, input_server_status[['host_name', 'cpu']], left_on='host_name_U', right_on='host_name', how='left')
    
    temp_input_looplist = A2[['host_name_x', 'machine_type_x', 'model_filename', 'cpu','cpu_x', 'cpu_y', 'amb_temp' ]]
    temp_input_looplist['lower_amb_temp'] = temp_input_looplist['amb_temp']
    temp_input_looplist['upper_amb_temp'] = temp_input_looplist['amb_temp']
    temp_input_looplist.rename(columns={'host_name_x': 'host_name', 'machine_type_x': 'machine_type', 'cpu_x':'lower_cpu', 'cpu_y':'upper_cpu', 'cpu':'middle_cpu', 'amb_temp':'middle_amb_temp'}, inplace=True)
    
    predictions_backtemp=[]
    for index, (i, row) in enumerate(temp_input_looplist.iterrows()):

        cur_host_name = row['host_name']
        cur_machine_type = row['machine_type']
        cur_modle_name = row['model_filename']
        
        if pd.isna(cur_modle_name):
            cur_modle_name = 'catboost_ew1-12'
        
        cur_row_data = pd.DataFrame([row[['lower_cpu', 'upper_cpu', 'middle_cpu', 'lower_amb_temp', 'upper_amb_temp', 'middle_amb_temp']]])
        
        # Use the preloaded model
        cur_load_backtemp_model = model_dict[cur_modle_name]
        
        # Predict using the model
        prediction = cur_load_backtemp_model.predict(cur_row_data)[0]
 
        # Append the prediction details to the list
        predictions_backtemp.append({
        'host_name': cur_host_name,
        'machine_type': cur_machine_type,
        'prediction': prediction
        })        
        
    # Convert the list of dictionaries to a dataframe
    predictions_backtemp_df = pd.DataFrame(predictions_backtemp)

    return predictions_backtemp_df

        
def compute_penalty(back_planetemp_values, upper_threshold, lower_threshold):
    x = back_planetemp_values.prediction
    max_x = max(x)
    sum_x = sum(x)
    
    # Calculate the number of x >= upper_threshold
    num_above_upper_threshold = sum(xi >= upper_threshold for xi in x)
    
    # Calculate the number of lower_threshold <= x < upper_threshold
    num_within_thresholds = sum(lower_threshold <= xi < upper_threshold for xi in x)

    # Compute the penalty
    if max_x >= upper_threshold:
        penalty = 100000
    elif lower_threshold <= max_x < upper_threshold:
        penalty = (sum_x / 1000) ** 2
    else:
        penalty = 0
    
    return penalty, num_above_upper_threshold, num_within_thresholds


def process_server_data(target_DC_usage, fan_speed_indoor, temp_supply_air,  rand_adj=2):
      
    # get the dp
    input_speed = fan_speed_indoor
    rslt_server_df = find_closest_speed(dp_server_ref_table, input_speed)
    server_data = rslt_server_df
    
    server_data.rename(columns={'mean_dp':'dp'}, inplace=True)
    
    server_data = server_data.merge(sever_list[['host_name', 'machine_type']], on='host_name', how='left')
    
    # get the amb_temp
    server_data['amb_temp'] = temp_supply_air

    # get cpu ratio
    total_servers = server_data.shape[0]
    total_tasks_capacity = total_servers * 96
    num_target_tasks = round(total_tasks_capacity * target_DC_usage / 100)

    tasks_per_server = num_target_tasks // total_servers
    tasks_extra = num_target_tasks % total_servers  

    tasks_distribution = [tasks_per_server + (1 if i < tasks_extra else 0) for i in range(total_servers)]
    tasks_distribution_cpu_pect = [task / 96 * 100 for task in tasks_distribution]
    
    random_adjustments = np.random.uniform(-rand_adj, rand_adj, size=len(tasks_distribution_cpu_pect))
    tasks_distribution_cpu_pect_adj = [ratio + adjustment for ratio, adjustment in zip(tasks_distribution_cpu_pect, random_adjustments)]

    server_data['cpu'] = tasks_distribution_cpu_pect_adj
    #  if there is negative values, replace it with 0.5
    server_data['cpu'] = server_data['cpu'].apply(lambda x: x if x >= 0 else 0.5) 


    # predict power
    input_data_tmp = server_data[['machine_type', 'cpu', 'amb_temp', 'dp']]
    
    # predict lenovo
    input_data_lenovo = input_data_tmp[input_data_tmp['machine_type'] == 'e_Lenovo']
    input_data_lenovo = input_data_lenovo.drop(columns='machine_type')
    pred_PW_lenovo = model_eLenovo_load.predict(input_data_lenovo)

    # predict supermicro
    input_data_supermicro = input_data_tmp[input_data_tmp['machine_type'] == 'e_Supermicro']
    input_data_supermicro = input_data_supermicro.drop(columns='machine_type')
    
    pred_PW_supermicro = model_esupermicro_load.predict(input_data_supermicro)

    return pred_PW_lenovo, pred_PW_supermicro, server_data

def process_server_data_WAO(target_DC_usage, fan_speed_indoor, temp_supply_air,  rand_adj=2):
      
    # ----- get the dp
    input_speed = fan_speed_indoor
    rslt_server_df = find_closest_speed(dp_server_ref_table, input_speed)
    server_data = rslt_server_df
    
    # get the DP for all servers
    server_data.rename(columns={'mean_dp':'dp'}, inplace=True)
    
    server_data = server_data.merge(sever_list[['host_name', 'machine_type']], on='host_name', how='left')
    
    # get the amb_temp (Now has DP and ambt_emp)
    server_data['amb_temp'] = temp_supply_air

    # get cpu ratio
    total_servers = server_data.shape[0]
    total_tasks_capacity = total_servers * 96
    num_target_tasks = round(total_tasks_capacity * target_DC_usage / 100)

    # ------(1) predict base power
    server_data['cpu'] = 0.1
    input_data_tmp = server_data[['machine_type', 'cpu', 'amb_temp', 'dp']]
    
    # predict base power  
    all_predictions_basePW =[]
    for idx, row in server_data.iterrows():
        # Get current amb_temp and dp values from the row
        cur_ambtemp = row['amb_temp']
        cur_dp = row['dp']
        cur_cpu = row['cpu']
        cur_machine = row['machine_type']
        
        # Create predict_loop_df for the current row
        input_data_basePW = pd.DataFrame({
            'cpu': [cur_cpu],      
            'amb_temp': [cur_ambtemp] ,    
            'dp': [cur_dp]                
        })
        
        # Make power prediction and append to pred_basePW
        if cur_machine == 'e_Supermicro':
            all_predictions_basePW.append(model_esupermicro_load.predict(input_data_basePW)[0])
        else:
            all_predictions_basePW.append(model_eLenovo_load.predict(input_data_basePW)[0])
      

    # --- (2) get core increase power
   # List to hold adjusted predictions for each row in server_data
    all_predictions_adjusted = []
    
    # Loop through each row in server_data
    for idx, row in server_data.iterrows():
        # Get current amb_temp and dp values from the row
        cur_ambtemp = row['amb_temp']
        cur_dp = row['dp']
        cur_machine = row['machine_type']
        
        # Create predict_loop_df for the current row
        predict_loop_df = pd.DataFrame({
            'cpu': range(1, 97),      # 1 to 96
            'amb_temp': [cur_ambtemp] * 96,     # Repeating cur_ambtemp
            'dp': [cur_dp] * 96                 # Repeating cur_dp
        })
        
        # Make power prediction using predict_loop_df
        if cur_machine=='e_Supermicro':
            pred_PW_table = model_esupermicro_load.predict(predict_loop_df)
        else:
            pred_PW_table = model_eLenovo_load.predict(predict_loop_df)
            
        # Initialize pred_PW_supermicro_table_adj as a copy of the predictions
        pred_PW_table_adj = pred_PW_table.copy()
        
        # Adjust to ensure non-decreasing sequence
        for i in range(1, len(pred_PW_table_adj)):
            pred_PW_table_adj[i] = max(pred_PW_table_adj[i], pred_PW_table_adj[i - 1])
        
        # Append the adjusted predictions to the list
        all_predictions_adjusted.append(pred_PW_table_adj)
    
    # Convert all_predictions_adjusted to a DataFrame if needed
    all_predictions_adjusted_df = pd.DataFrame(all_predictions_adjusted).T  # Transpose to match original shape
    # all_predictions_adjusted_df.columns = [f'row_{i}' for i in range(len(server_data))]
    
    # Use the index of server_data as column names for all_predictions_adjusted_df
    all_predictions_adjusted_df.columns = server_data.index

    # add the based power to the matrix ==> this will make the based power in the index [0]
    all_predictions_adjusted_df.loc[-1] = all_predictions_basePW  # Temporary row at index -1
    all_predictions_adjusted_df.index = all_predictions_adjusted_df.index + 1  # Shift index by 1
    all_predictions_adjusted_df = all_predictions_adjusted_df.sort_index()  # Sort index to reset order
   
    all_predictions_adjusted_df_sort = all_predictions_adjusted_df.apply(lambda x: x.sort_values(ascending=True).values)

    
      
    # Generate the new dataframe by subtracting each row from the previous row 
    all_predictions_adjusted_df_PWincrease = all_predictions_adjusted_df_sort.diff().fillna(0)

    # (4) use uniform
    basePW_total = all_predictions_adjusted_df.iloc[0].sum() / 1000
    
    # ** get the PW increase table, this is the core for doing algorithm caculation 
    all_predictions_adjusted_df_PWincrease_nobasePW=all_predictions_adjusted_df_PWincrease.drop(index=0).reset_index(drop=True)   

    # Calculate uniform PW --> using rolling sums  increased PW
    all_predictions_adjusted_df_PWincrease_nobase_flate = all_predictions_adjusted_df_PWincrease_nobasePW.values.flatten()
    PW_unifrom_increase_total = sum(all_predictions_adjusted_df_PWincrease_nobase_flate[:num_target_tasks]) / 1000
    PW_total_unfirom = basePW_total + PW_unifrom_increase_total
    

    # use (5) WAO
    # Create the predefined DataFrame df_WAO_order   
    negative_numbers_count = (all_predictions_adjusted_df_PWincrease < 0).sum().sum()
    
    all_predictions_adjusted_df_PWincrease_proc = all_predictions_adjusted_df_PWincrease.copy()
    df_WAO_order = pd.DataFrame(columns=['idx_row', 'server', 'value'])
    
    # get the WAO and unifrom's reslt
    df_WAO_order, df_PW_increase = move_min_values(all_predictions_adjusted_df_PWincrease_proc, df_WAO_order, num_target_tasks)
    
    PW_WAO_increase_total = df_WAO_order['value'].sum()/1000
    PW_total_WAO = basePW_total + PW_WAO_increase_total
    

    return PW_total_unfirom, PW_total_WAO, server_data


def objective(trial, cur_workload, cur_temp_outdoor, cur_humid_outdoor):
    global pred_PW_all_Server_hist, Control_FanSpeed_hist, Control_ChillerWaterTemp_hist, pred_TempSupplyAir_hist, pred_PW_HVAC_hist, pred_PW_HVAC_ref_hist, pred_PW_all_Lenovo_hist, pred_PW_all_Supermicro_hist, backplane_temp_lenovo_hist, backplane_temp_supermicro_hist 
    global all_trials_df_hist_list  # Declare the global variable
    global pred_PW_unifrom_hist, pred_PW_WAO_hist
    
    param = {
        'temp_chiller_water': trial.suggest_int('temp_chiller_water', 3, 32, step=1), # set as init
        'fan_speed_indoor_1': trial.suggest_int('fan_speed_indoor_1', 950, 1400, step=20), # set as init
        # 'fan_speed_outdoor_1': 600, # fixed
        # 'pump_inv_freq_1': 40, # fixed
        'humid_outdoor': cur_humid_outdoor, # real dynamic
        'temp_outdoor': cur_temp_outdoor, # real dynamic
        # 'dp_hvac_1': 75, # set as init
        # 'power_total_hvac':3 # set as init
    }

    # making control here
    control_FanSpeed = param['fan_speed_indoor_1']
    control_TempChillerWater = param['temp_chiller_water']    
    
    # --[Start from control] predict temp_supply_air, Predict using the loaded model,  # e.g   'temp_supply_air': 20,
    input_df_hvac_SupplyAirTemp = pd.DataFrame([param])  # Create a DataFrame with one row of parameters
    
    #update HVAC's DP
    rslt_hvac_dp = find_nearest_dp_hvac(control_FanSpeed)  
    input_df_hvac_SupplyAirTemp["dp_hvac_1"] = rslt_hvac_dp  
    
    Pred_temp_supply_air = round(model_HVAC_AirTemp_load.predict(input_df_hvac_SupplyAirTemp)[0])  # Assume model returns an array, take the first value

    
    # based on Kagoya's log history
    cur_workload_input = cur_workload 
    
    
    # ----[get Server's PW_uniform]
    pred_pw_lenovo_array , pred_pw_supermicro_array,  server_status = process_server_data(     
        cur_workload_input,
        control_FanSpeed,
        Pred_temp_supply_air
    )
    
    # check individual server type
    pred_PW_Lenovo_server = (np.sum(pred_pw_lenovo_array) / 1000)  # for Lenovo
    pred_PW_Supermicro_server = (np.sum(pred_pw_supermicro_array) / 1000)  # for supermicro
    # pred_PW_both_server = pred_PW_Lenovo_server + pred_PW_Supermicro_server use WAO's approach
    
    
    # **** key ----[get Server's PW with WAO]
    PW_total_unfirom, PW_total_WAO,   server_status = process_server_data_WAO(     
        cur_workload_input,
        control_FanSpeed,
        Pred_temp_supply_air
    )
    pred_PW_array = np.concatenate((pred_pw_lenovo_array, pred_pw_supermicro_array), axis=0)
    # pred_PW_both_server = np.sum(pred_PW_array) / 1000
    # pred_PW_both_server = pred_PW_both_server - 0.5 # adjustment
    
    # check individual server
    # pred_PW_Lenovo_server = (np.sum(pred_pw_lenovo_array) / 1000)  # for Lenovo
    # pred_PW_Supermicro_server = (np.sum(pred_pw_supermicro_array) / 1000)  # for supermicro
    # PW_total_unfirom = 10
    pred_PW_both_server = PW_total_unfirom #pred_PW_Lenovo_server + pred_PW_Supermicro_server
 
    # PW_total_WAO = 8
    
    
    #  ---[HVAC power]----
    param['power_server'] = pred_PW_both_server # update server power
    # Convert parameters to a DataFrame for prediction
    input_df_hvac_pw = pd.DataFrame([param])  # Create a DataFrame with one row of parameters
    
    # 7 input to get the hvac power
    # [["temp_chiller_water", "fan_speed_indoor_1",  "humid_outdoor",  "dp_hvac_1", "temp_outdoor",  "temp_supply_air", "power_server", "power_total_hvac" ]]

    # add predicted HVAC temp_supply_air
    input_df_hvac_pw["temp_supply_air"] = Pred_temp_supply_air  
    
    #update HVAC's DP
    input_df_hvac_pw["dp_hvac_1"] = rslt_hvac_dp  
    
    # input_df_hvac_pw = input_df_hvac_pw.drop('power_total_hvac', axis=1)
    # this is the predicted optimal HVAC control's power
    PW_hvac = model_HVAC.predict(input_df_hvac_pw)[0]  # Assume model returns an array, take the first value

    input_df_hvac_pw_ref = input_df_hvac_pw.copy()
    input_df_hvac_pw_ref['temp_chiller_water'] = init_temp_chiller_water
    input_df_hvac_pw_ref['fan_speed_indoor_1'] = init_fan_speed
    
    rslt_hvac_dp_ref = find_nearest_dp_hvac(init_fan_speed)  
    input_df_hvac_pw_ref["dp_hvac_1"] = rslt_hvac_dp_ref  
    
    
    param_hvac_ref = {
        'temp_chiller_water': init_temp_chiller_water, # set as init
        'fan_speed_indoor_1': init_fan_speed, # set as init
        # 'fan_speed_outdoor_1': 600, # fixed
        # 'pump_inv_freq_1': 40, # fixed
        'humid_outdoor': cur_humid_outdoor, # real dynamic
        'temp_outdoor': cur_temp_outdoor, # real dynamic
        'dp_hvac_1': rslt_hvac_dp_ref, # set as init
    }   
    input_df_hvac_SupplyAirTemp_ref = pd.DataFrame([param_hvac_ref])
    Pred_temp_supply_air = round(model_HVAC_AirTemp_load.predict(input_df_hvac_SupplyAirTemp_ref)[0])
    input_df_hvac_pw_ref['temp_supply_air'] = Pred_temp_supply_air
    
    # this is the predicted reference HVAC control's power
    PW_hvac_ref = model_HVAC.predict(input_df_hvac_pw_ref)[0]

    # make records for [total server PW] and [hvac's PW]
    pred_PW_all_Lenovo_hist.append(pred_PW_Lenovo_server)  
    pred_PW_all_Supermicro_hist.append(pred_PW_Supermicro_server) 
    pred_PW_all_Server_hist.append(pred_PW_both_server)
    pred_PW_HVAC_hist.append(PW_hvac)  # record results of the trial
    pred_PW_HVAC_ref_hist.append(PW_hvac_ref)  # record results of the trial
    pred_PW_unifrom_hist.append(PW_total_unfirom)
    pred_PW_WAO_hist.append(PW_total_WAO)
    
    Control_FanSpeed_hist.append(control_FanSpeed)
    Control_ChillerWaterTemp_hist.append(control_TempChillerWater)
    pred_TempSupplyAir_hist.append(Pred_temp_supply_air) # record results of prediced temp supply air

    # Calculate the score based on the model output (this is also the record of total DC PW)
        
    # ---[hotspot pentley]----
    #(1) get the backplate temp using backplate temp models
    back_planetemp_values = compute_backplane_temp(server_status)
    
    back_planetemp_values_lenovo = back_planetemp_values.loc[back_planetemp_values['machine_type']=='e_Lenovo'].reset_index(drop=True)
    pentley_value_lenovo, num_above_upper_threshold_lenovo, num_within_thresholds_lenovo = compute_penalty(back_planetemp_values_lenovo, upper_threshold=55, lower_threshold=50)
  
    back_planetemp_values_supermicro = back_planetemp_values.loc[back_planetemp_values['machine_type']=='e_Supermicro'].reset_index(drop=True)
    pentley_value_supermicro, num_above_upper_threshold_supermicor, num_within_thresholds_supermicro = compute_penalty(back_planetemp_values_supermicro, upper_threshold=45, lower_threshold=40)

    backplane_temp_lenovo_hist.append((num_above_upper_threshold_lenovo, num_within_thresholds_lenovo))
    backplane_temp_supermicro_hist.append((num_above_upper_threshold_supermicor, num_within_thresholds_supermicro))

    score = PW_hvac + pred_PW_both_server + pentley_value_lenovo + pentley_value_supermicro

    return score




def calculate_savings(original, optimized):
    original = np.array(original)
    
    savings = (original - optimized) / original * 100
    max_saving = savings.max()
    min_saving = savings.min()
    # total_saving_rate = np.sum(original - optimized)
    
    
    kwh_convert = (original - optimized)/6
    kwh_saving = kwh_convert.sum()
    
    electric_eng_saving_rate = kwh_saving / np.sum(original/6) * 100
    
    return max_saving, min_saving, kwh_saving, electric_eng_saving_rate

def calculate_increase(original, optimized):
    original = np.array(original)
    
    increase = (optimized-original) / optimized * 100
    max_increase = increase.max()
    min_increase = increase.min()
    
    kwh_convert = (optimized - original )/6
    kwh_increase = kwh_convert.sum()
    
    electric_eng_increase_rate = kwh_increase / np.sum(original/6) * 100
    
    return max_increase, min_increase, kwh_increase, electric_eng_increase_rate



# Function to plot the progress of a single trial
def plot_trial_progress(trial_index, trial_progress_df):
    plt.figure(figsize=(10, 6), dpi=300)  # Set figure size
    plt.plot(trial_progress_df['number'], trial_progress_df['value'], marker='o', linestyle='-', color='blue')
    plt.title(f' Optimal parameter searching progress for Trial {trial_index}')
    # plt.title(f' Optimal parameter searching progress')
    plt.xlabel('Trial Number')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()
    
def move_min_values(df_source, df_target, num_moves):
    # Convert DataFrame to NumPy array of type float to allow NaN values
    data_array = df_source.to_numpy(dtype=float)
    num_rows, num_cols = data_array.shape
    col_names = df_source.columns.to_list()
    idx_array = df_source.index.to_numpy()
    
    # Initialize pointers for each column
    pointers = np.zeros(num_cols, dtype=int)
    
    for _ in range(num_moves):
        # Prepare arrays to hold current values and indices
        current_values = np.full(num_cols, np.nan)
        current_indices = np.full(num_cols, -1, dtype=int)
        
        # Find the current value for each column
        for col_idx in range(num_cols):
            ptr = pointers[col_idx]
            # Skip NaN values
            while ptr < num_rows and np.isnan(data_array[ptr, col_idx]):
                ptr += 1
            if ptr < num_rows:
                current_values[col_idx] = data_array[ptr, col_idx]
                current_indices[col_idx] = idx_array[ptr]
                pointers[col_idx] = ptr  # Update pointer
            else:
                pointers[col_idx] = num_rows  # Mark as exhausted
            
        # Filter out columns that are exhausted
        valid_mask = ~np.isnan(current_values)
        if not np.any(valid_mask):
            break  # All columns are exhausted
        
        # Find the column with the minimum value
        min_col_idx = np.argmin(current_values[valid_mask])
        # Map back to original indices
        valid_col_indices = np.where(valid_mask)[0]
        min_col = valid_col_indices[min_col_idx]
        min_value = current_values[min_col]
        min_idx = current_indices[min_col]
        min_col_name = col_names[min_col]
        
        # Append the found value to df_target
        new_row = pd.DataFrame({
            'idx_row': [min_idx],
            'server': [min_col_name],
            'value': [min_value]
        })
        df_target = pd.concat([df_target, new_row], ignore_index=True)
        
        # Move the pointer for that column to the next value
        pointers[min_col] += 1
        # Set the moved value to NaN in the data_array
        data_array[int(pointers[min_col]-1), min_col] = np.nan
    
    # Update df_source with the modified data_array
    df_source.loc[:, :] = data_array
    
    return df_target, df_source
    



#%% run sequential : Create a study object and optimize the objective
best_results = []
List_all_HVAC_WAO_df = []  # List to store each DataFrame
start_time = time.time()

for idx in range(len(df_workload_kagoya)):
    print(idx)
        
    cur_temp_outdoor = df_HVAC.loc[idx]["temp_outdoor"]
    cur_humid_outdoor = df_HVAC.loc[idx]["humid_outdoor"]
    cur_workload = df_workload_kagoya.loc[idx]["load"] 
    
    # Reset global variables
    pred_PW_all_Server_hist = []
    pred_PW_HVAC_hist = []
    pred_PW_HVAC_ref_hist = []
    pred_PW_all_Lenovo_hist = []
    pred_PW_all_Supermicro_hist = []
    pred_PW_unifrom_hist =[]
    pred_PW_WAO_hist = []
    
    Control_FanSpeed_hist=[]
    Control_ChillerWaterTemp_hist=[]
    pred_TempSupplyAir_hist= []
     
    # start optimize
    study = optuna.create_study(direction='minimize')  # Assuming the goal is to minimize the score
    study.optimize(lambda trial: objective(trial, cur_workload, cur_temp_outdoor, cur_humid_outdoor), n_trials=50)

    # analyze optimize result
    all_trials_df = study.trials_dataframe()
    
    best_trial = study.best_trial
    trail_id_best = best_trial.number  # or all_trials_df["value"].idxmin()
    
    best_server_Pw = pred_PW_all_Server_hist[trail_id_best]
    best_server_PW_lenovo = pred_PW_all_Lenovo_hist[trail_id_best]
    best_server_PW_supermicor = pred_PW_all_Supermicro_hist[trail_id_best]
    
    
    best_HVAC_Pw = pred_PW_HVAC_hist[trail_id_best]
    best_total_DC_Pw = best_server_Pw + best_HVAC_Pw
    
    best_controlfan_speed = Control_FanSpeed_hist[trail_id_best]
    best_controlChillerTemp = Control_ChillerWaterTemp_hist[trail_id_best]
    best_predAirTemp = pred_TempSupplyAir_hist[trail_id_best]

    best_results.append({
        'trial': idx,
        'control_FanSpeed': best_controlfan_speed,
        'control_chillerTemp': best_controlChillerTemp,
        'best_predAirTemp': best_predAirTemp,
        'best_server_Pw': best_server_Pw,
        'best_server_Pw_lenovo': best_server_PW_lenovo,
        'best_server_Pw_supermicro': best_server_PW_supermicor,
        'best_HVAC_Pw': best_HVAC_Pw,
        'best_total_DC_Pw': best_total_DC_Pw
    })
    
    # Now, collect HVAC + WAO method's result
    df_WAO_HVAC_effect = pd.DataFrame({
        'cpu':cur_workload,
        'control_ChillerTemp':best_controlChillerTemp,
        'control_FanSpeed':Control_FanSpeed_hist,
        'PW_HVAC': pred_PW_HVAC_hist,
        'PW_HVAC_ref': pred_PW_HVAC_ref_hist,
        'PW_uniform': pred_PW_unifrom_hist,
        'PW_WAO': pred_PW_WAO_hist
    })
    
    
    # Calculating other columns
    df_WAO_HVAC_effect['WAO_saving_value'] = df_WAO_HVAC_effect['PW_uniform']-df_WAO_HVAC_effect['PW_WAO']
    df_WAO_HVAC_effect['WAO_saving_ratio'] = df_WAO_HVAC_effect['WAO_saving_value'] / df_WAO_HVAC_effect['PW_uniform']
    
    List_all_HVAC_WAO_df.append(df_WAO_HVAC_effect)
    

# # Now best_results contains the best results for each loop iteration
best_results_df = pd.DataFrame(best_results)
# best_results_df.to_csv('Optuna_kagoya_best_results_vhotspot.csv', index=False)
# print(best_results_df)

# # # End the timer
end_time = time.time()
# Calculate the processing time
processing_time = round(end_time - start_time,3)
print(f"\nOptimization took {processing_time} seconds.")

# save List_all_HVAC_WAO_df (to pickle)
# Save the list to a pickle file
with open('test_result.pkl', 'wb') as file:
    pickle.dump(List_all_HVAC_WAO_df, file)

