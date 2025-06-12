# Functions to retrieve latest (based on analysistime) METAR observations from AQU service
import numpy as np
import pandas as pd
import requests
import io
import time
from datetime import datetime, timedelta

def metar_from_aqu_try(start, end, icao, param, paramName):
    """
    returns time series of observations from smartmet server
    """
    proxies = {'https':'http://wwwproxy.fmi.fi:8080'}
    url = 'https://lentosaa.fmi.fi/nordictafverif/tafverif/aqu.php?icaoId=\'{icao}\'&startTime={start}&endTime={end}&paramId={param}&messageTypeId=1,8&timeStep=30'.format(icao=icao,start=start,end=end,param=str(param))
    count=0
    while (count<10):
        try:
            response = requests.get(url, proxies=proxies, timeout=15).text
            return pd.read_csv(io.StringIO(response), names=['time',paramName], sep=';')
        except requests.exceptions.Timeout:
            if (count<5):
                #print(f"Request timed out after {5} seconds. Wait 5 seconds. ({count})")
                time.sleep(5)
            else:
                print(icao)
                print(f"Request timed out after {15} seconds. Wait 15 seconds. ({count})")
                time.sleep(15)
            count+=1
            continue
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    return None

def retrieve_latest_metar_observations(icao, analysistime):
    #Start time is 1 hour before analysistime
    analysistime_dt = datetime.strptime(analysistime, '%Y%m%d%H%M%S')
    start_time = analysistime_dt - timedelta(hours=1)
    start = start_time.strftime('%Y%m%d%H%M')
    #End time is 3 hour after analysistime
    end_time = analysistime_dt + timedelta(hours=3)
    end = end_time.strftime('%Y%m%d%H%M')
    print(f"Retrieving METAR observations for {icao} from {start} to {end}")
    #Retrieve METAR observations from AQU service for parameters 1,4,10,20,513,515,517 and paramName 'p', 't', 'td', 'wd', 'cbase', 'vis' and 'ws'
    metar_obs = metar_from_aqu_try(start, end, icao, param=1, paramName="p")
    param_dict = {4: 't', 10: 'td', 20: 'wd', 501: 'cc1', 502: 'cbase1', 513: 'cbase', 515: 'vis', 517: 'ws'}
    for param, paramName in param_dict.items():
            metar_obs_param = metar_from_aqu_try(start, end, icao, param=param, paramName=paramName)
            if metar_obs_param is not None:
                metar_obs = pd.merge(metar_obs, metar_obs_param, on='time', how='outer')
            else:
                continue
    #Do all the necessary modifications to the METAR observations        
    #Convert cbase unit hft to meters and replace missing values with 7500
    metar_obs['cbase'] = round(metar_obs['cbase'] * 0.3048 * 100,2)
    metar_obs['cbase'] = metar_obs['cbase'].replace(np.nan,7500)

    #Replace missing first cloud layer cloud cover (cc1) with value 0
    metar_obs['cc1'] = metar_obs['cc1'].replace(np.nan,0)
    #Convert cbase1 unit hft to meters and replace missing values with 7500
    metar_obs['cbase1'] = round(metar_obs['cbase1'] * 0.3048 * 100,2)
    metar_obs['cbase1'] = metar_obs['cbase1'].replace(np.nan,7500) 

    #Check the cases where there is no cloud, but vertical visibility is low
    #If cbase is lower than cbase1, set cc1 to 9 and cbase1 to cbase
    metar_obs['cc1'] = np.where((metar_obs['cbase'] < metar_obs['cbase1']), 9, metar_obs['cc1'])
    metar_obs['cbase1'] = np.where((metar_obs['cbase'] < metar_obs['cbase1']), metar_obs['cbase'], metar_obs['cbase1'])
            
    #Replace missing wd value with -1
    metar_obs['wd'] = metar_obs['wd'].replace(np.nan,-1)

    #If vis is missing, replace also other modified cloud parameter values with np.nan 
    metar_obs['cbase'] = np.where(metar_obs['vis'].isnull(), np.nan, metar_obs['cbase'])
    metar_obs['cbase1'] = np.where(metar_obs['vis'].isnull(), np.nan, metar_obs['cbase1'])
    metar_obs['cc1'] = np.where(metar_obs['vis'].isnull(), np.nan, metar_obs['cc1'])

    #If ws is missing, replace also wind direction value with np.nan 
    metar_obs['wd'] = np.where(metar_obs['ws'].isnull(), np.nan, metar_obs['wd'])

    #Order columns
    metar_obs = metar_obs[['time', 'p', 't', 'td', 'ws', 'wd', 'cc1', 'cbase1', 'cbase', 'vis']]
    
    return metar_obs

def latest_metar_observations_features(icao, analysistime):
    metar_observations = retrieve_latest_metar_observations(icao, analysistime)
    all_col_names = []
    for param in ['p', 't', 'td', 'ws', 'wd', 'cc1', 'cbase1', 'cbase', 'vis']:
        param_col_names = [f"{param}_{i+1}" for i in range(6)]
        all_col_names.extend(param_col_names)
    metar_obs_array = pd.DataFrame(columns=all_col_names,index=None)
    all_metar_values = []
    for param in ['p', 't', 'td', 'ws', 'wd', 'cc1', 'cbase1', 'cbase', 'vis']:
        metar_obs_param = metar_observations[param]
        #Remove missing values
        metar_obs_param = metar_obs_param[metar_obs_param.notnull()]
        #Check that len of metar_obs_param is at least 6 and take the last 6 values in reverse order
        if len(metar_obs_param) >= 6:
            metar_obs_param = metar_obs_param.iloc[::-1].head(6)
        else:
            #six Nan values
            metar_obs_param = pd.Series([np.nan]*6)
        #Add the values to the all_metar_values list
        all_metar_values.extend(metar_obs_param.tolist())
    metar_obs_array.loc[len(metar_obs_array)] = all_metar_values

    return metar_obs_array

#Example of use   
#icao = 'EFHK'  # Example ICAO code
#analysistime = '202506060600'  # Example analysistime in format YYYYMMDDHHMMSS
#latest_metar_features = latest_metar_observations_features(icao, analysistime)
