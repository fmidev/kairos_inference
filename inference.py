import numpy as np
import sys
import argparse
import pandas as pd
import gridpp
import time
import xgboost as xgb
from fileutils import read_grib_time, write_grib
from metarutils import latest_metar_observations_features

def parse_kv(kv):
    """Parse a key=value string into a dictionary."""
    if kv is None:
        return None
    if isinstance(kv, str):
        kv = [kv]
    d = {}
    for item in kv:
        if item == "None":
            continue
        k, v = item.split("=")
        try:
            v = float(v)
        except ValueError:
            pass
        d[k] = v
    return d

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--clb", action="store", type=str, required=True)
    parser.add_argument("--wd", action="store", type=str, required=True)
    parser.add_argument("--ws", action="store", type=str, required=True)
    parser.add_argument("--t2m", action="store", type=str, required=True)
    parser.add_argument("--t10m", action="store", type=str, required=True)
    parser.add_argument("--t0m", action="store", type=str, required=True)
    parser.add_argument("--ppa", action="store", type=str, required=True)
    parser.add_argument("--vis", action="store", type=str, required=True)
    parser.add_argument("--rh2m", action="store", type=str, required=True)
    parser.add_argument("--rh10m", action="store", type=str, required=True)
    parser.add_argument("--bld", action="store", type=str, required=True)
    parser.add_argument("--rr", action="store", type=str, required=True)
    parser.add_argument("--snr", action="store", type=str, required=True)
    parser.add_argument("--cc", action="store", type=str, required=True)
    parser.add_argument("--lcc", action="store", type=str, required=True)
    parser.add_argument("--rnetsw", action="store", type=str, required=True)
    parser.add_argument("--rnetlw", action="store", type=str, required=True)
    parser.add_argument("--model_vis_0", action="store", type=str, required=True)
    parser.add_argument("--model_vis_1", action="store", type=str, required=True)
    parser.add_argument("--model_vis_2", action="store", type=str, required=True)
    parser.add_argument("--model_cbase_0_36", action="store", type=str, required=True)
    parser.add_argument("--model_cbase_4_9", action="store", type=str, required=True)
    parser.add_argument("--producer_id", action="store", type=int, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    #parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--grib_options", action="store", nargs="+", metavar="KEY=VALUE", type=str, default=None, required=False)
    args = parser.parse_args()
    allowed_params = ["visibility", "cldbase"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)
    args.grib_options = parse_kv(args.grib_options)
    return args

def read_grid(args):
    """Top function to read "all" gridded data"""

    lons, lats, clb, analysistime, forecasttime = read_grib_time(args.clb, True)
    _, _, wd, _, _ = read_grib_time(args.wd, False)
    _, _, ws, _, _ = read_grib_time(args.ws, False)
    _, _, t2m, _, _ = read_grib_time(args.t2m, False)
    _, _, t10m, _, _ = read_grib_time(args.t10m, False)
    _, _, t0m, _, _ = read_grib_time(args.t0m, False)
    _, _, ppa, _, _ = read_grib_time(args.ppa, False)
    _, _, vis, _, _ = read_grib_time(args.vis, False)
    _, _, rh2m, _, _ = read_grib_time(args.rh2m, False)
    _, _, rh10m, _, _ = read_grib_time(args.rh10m, False)
    _, _, bld, _, _ = read_grib_time(args.bld, False)
    _, _, rr, _, _ = read_grib_time(args.rr, False)
    _, _, snr, _, _ = read_grib_time(args.snr, False)
    _, _, cc, _, _ = read_grib_time(args.cc, False)
    _, _, lcc, _, _ = read_grib_time(args.lcc, False)
    _, _, rnetsw, _, _ = read_grib_time(args.rnetsw, False)
    _, _, rnetlw, _, _ = read_grib_time(args.rnetlw, False)
    
    missing_data = 9999
    # check if any input grib_files contain missing data. If missing data then exit program
    all_input = {'clb': clb, 'wd': wd, 'ws': ws, 't2m': t2m, 't10m': t10m, 't0m': t0m, 'ppa': ppa, 'vis': vis,
                    'rh2m': rh2m, 'rh10m': rh10m, 'bld': bld, 'rr': rr, 'snr': snr, 'cc': cc, 'lcc': lcc, 'rnetsw': rnetsw, 'rnetlw': rnetlw}
    for name, arr in all_input.items():
        if missing_data in arr:
            print(f"Missing data found in {name}")
            #exit("Aborting program due to missing data.")

    # Change parameter units:
    ppa = ppa / 100
    #cl = np.around(cl / 0.125, 0)
    return analysistime, forecasttime, lons, lats, clb, wd, ws, t2m, t10m, t0m, ppa, vis, rh2m, rh10m, bld, rr, snr, cc, lcc, rnetsw, rnetlw

def point_interpolate(analysistime, forecasttime, clb, wd, ws, t2m, t10m, t0m, ppa, vis, rh2m, rh10m, bld, rr, snr, cc, lcc, rnetsw, rnetlw, grid, points, idxs):
    """Interpolate the data to the points and store to dataframe"""
    leadtime = (forecasttime - analysistime) / pd.Timedelta(hours=1)
    # Four closest points for ceiling
    #print("Indices of the closest points:", idxs)
    rows, cols = idxs[:, 0], idxs[:, 1]
    ceiling = []
    for i in range(len(leadtime)):
        tmp =  clb[i, rows, cols] # four value
        # tmp values that are 9999 are missing data and converted to nan
        tmp = np.where(tmp == 9999, np.nan, tmp)
        # get the nan.mean of the four values
        # guard against empty (all-NaN) slice
        if np.all(np.isnan(tmp)):
            tmp = np.nan
        else:
            tmp = np.nanmean(tmp)
        #print("Ceiling value at the point:", tmp)
        #clip the ceiling value to 7500
        tmp = np.clip(tmp, 0, 7500)  # clip to 0-7500
        # convert nan to 7500
        if np.isnan(tmp):
            tmp = 7500
        ceiling.append(tmp)
    #print("Indices of 4 closest grid cells:", idxs)
   
    #print("Leadtime in hours:", leadtime)
    #print("Values at those cells:", ceiling)
    df_data = pd.DataFrame()
    df_data["leadtime"] = leadtime.astype(int)
    df_data["validdate"] = forecasttime #analysistime + pd.to_timedelta(df_data["leadtime"], unit='h')
    df_data["ceiling"] = ceiling
    df_data["clb"] = gridpp.bilinear(grid, points, clb)
    df_data["wd"] = gridpp.bilinear(grid, points, wd)
    df_data["ws"] = gridpp.bilinear(grid, points, ws)
    df_data["t2m"] = gridpp.bilinear(grid, points, t2m)
    df_data["t10m"] = gridpp.bilinear(grid, points, t10m)
    df_data["t0m"] = gridpp.bilinear(grid, points, t0m)
    df_data["ppa"] = gridpp.bilinear(grid, points, ppa)
    df_data["vis"] = gridpp.bilinear(grid, points, vis)
    df_data["rh2m"] = gridpp.bilinear(grid, points, rh2m)
    df_data["rh10m"] = gridpp.bilinear(grid, points, rh10m)
    df_data["bld"] = gridpp.bilinear(grid, points, bld)
    df_data["rr"] = gridpp.bilinear(grid, points, rr)
    df_data["snr"] = gridpp.bilinear(grid, points, snr)
    df_data["cc"] = gridpp.bilinear(grid, points, cc)
    df_data["lcc"] = gridpp.bilinear(grid, points, lcc)
    df_data["rnetsw"] = gridpp.bilinear(grid, points, rnetsw)
    df_data["rnetlw"] = gridpp.bilinear(grid, points, rnetlw)    
    return df_data

def encode(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    return data

def preprocess_vis(df_data):
    """
    leadtime', 't0m', 't2m', 't65', 'rh2m', 'sp', 'lcc', 'visibility',
       'mld', 'rh65', 't_inv', 'snow_hourly', 'rain_hourly', 'nlwrs_hourly',
       'nswrs_hourly', 't2m_trend', 'visibility_trend', 'rh2m_trend', 'ws',
       'wd', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'obs_ws_t3h',
       'obs_cbase_t3h', 'obs_vis_t3h', 'ceiling',
       'obs_dp_diff'
       """
    # Select relevant columns for visibility
    df = df_data[['leadtime', 'validdate', 't0m', 't2m', 't10m', 'rh2m', 'ppa', 'lcc', 'vis', 'bld', 'rh10m', 
                  'snr', 'rr', 'rnetlw', 'rnetsw', 'ws', 'wd', 'ceiling', 'ws_1', 'cbase_1', 'vis_1', 't_1', 'td_1']]
    # Calculate sin and cos for month and hour
    df = df.assign(month=df.validdate.dt.month)
    df = df.assign(hour=df.validdate.dt.hour)
    df = encode(df, "month", 12)
    df = encode(df, "hour", 24)
    df["vis"] = df["vis"].clip(upper=10000)
    df["t2m_trend"] = df.t2m.diff().fillna(0)
    df["visibility_trend"] = df.vis.diff().fillna(0)
    df["rh2m_trend"] = df.rh2m.diff().fillna(0)
    df["rr"] = df.rr.diff().fillna(0)
    # Clip values under zero
    df["rr"] = df["rr"].clip(lower=0)
    df['obs_dp_diff'] = df['t_1'] - df['td_1']
    df = df.drop(["month", "hour"], axis=1)
    df["t_inv"] = df["t10m"] - df["t0m"] # ok
    # Rename columns for clarity
    df = df.rename(columns={
        't10m': 't65', 
        'ppa': 'sp', 
        'vis': 'visibility', 
        'bld': 'mld', 
        'rh10m': 'rh65', 
        'snr': 'snow_hourly', 
        'rr': 'rain_hourly', 
        'rnetlw': 'nlwrs_hourly', 
        'rnetsw': 'nswrs_hourly', 
        "ceiling": "ceiling",
        "ws_1": "obs_ws_t3h",
        "cbase_1": "obs_cbase_t3h",
        "vis_1": "obs_vis_t3h",
    })
    # Reorder columns for ML
    df = df[[
        'leadtime', 't0m', 't2m', 't65', 'rh2m', 'sp', 'lcc', 'visibility',
       'mld', 'rh65', 't_inv', 'snow_hourly', 'rain_hourly', 'nlwrs_hourly',
       'nswrs_hourly', 't2m_trend', 'visibility_trend', 'rh2m_trend', 'ws',
       'wd', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'obs_ws_t3h',
       'obs_cbase_t3h', 'obs_vis_t3h', 'ceiling',
       'obs_dp_diff'
    ]]
    # Remove leadtimes <= 3
    df = df[df['leadtime'] > 3]
    # Round everything to 2 decimals
    df = df.round(2)
    return df

def ml_forecast_vis(df, args):
    """
    Make a forecast with ML
    """
    fcst_probs = []
    start = [3, 6, 12, 18, 24, 33 ]
    vis_models = [args.model_vis_0, args.model_vis_1, args.model_vis_2]
    for i in range(2):
        # Load the model
        startd = start[i]
        endd = start[i+1]
        dfc = df[(df["leadtime"] > startd) & (df["leadtime"] <= endd)]
        model_path = vis_models[i] #f"xgb_vis_{i}_0625.json"
        bst = xgb.Booster()
        bst.load_model(model_path)
        # Predict using the model
        tmp = bst.predict(xgb.DMatrix(dfc))
        fcst_probs.append(tmp)
    # Stack lists
    fcst_probs = np.concatenate(fcst_probs, axis=0)
    # get the forecasted class
    fcst_vis = fcst_probs.argmax(axis=1) 
    # Second best if model is unsure about the class 9
    fcst_data = pd.DataFrame(fcst_probs, columns=[f"Class{i}" for i in range(fcst_probs.shape[1])])
    fcst_data["fcst_vis"] = fcst_vis  # Add predicted class
    # Find the second highest probability for each row
    fcst_data["second_best_class"] = fcst_data.drop(columns=["fcst_vis"]).apply(lambda row: row.nlargest(2).idxmin(), axis=1)
    fcst_data["second_best_class"] = fcst_data["second_best_class"].str.extract(r'(\d+)').astype(int)
    # Apply the rule: if predicted class is 8 and prob < 0.5, replace with second best
    mask = (fcst_data["fcst_vis"] == 8) & (fcst_data["Class8"] < 0.5)
    fcst_data.loc[mask, "fcst_vis"] =fcst_data.loc[mask, "second_best_class"]
    # create MEPS vis class for the rest of the leadtimes
    dfs = df[(df["leadtime"] > 12)].copy()    
    # Convert to visibility class
    bins = [0, 150, 350, 600, 800, 1500, 3000, 5000, 8000, np.inf]
    labels = [
        "<=150 m",
        "150-350 m",
        "350-600 m",
        "600-800 m",
        "800-1500 m",
        "1500-3000 m",
        "3000-5000 m",
        "5000-8000 m",
        ">8000 m",
    ]
    dfs["meps_cat"] = pd.cut(dfs["visibility"], bins=bins, labels=labels, right=True)
    dfs["meps_class"] = dfs["meps_cat"].cat.codes
    # Add obs visibility as the first value
    obs_vis = pd.cut(dfs["obs_vis_t3h"],bins=bins,labels=labels,right=True).cat.codes
    obs0 = obs_vis.iloc[0]
    tmp_vis = [obs0]  # Start with the observation visibility class
    # Extend the list with forecast visibility classes and MEPS visibility classes
    fcst_ser = fcst_data["fcst_vis"].tolist()
    meps_ser = dfs["meps_class"].tolist()
    tmp_vis.extend(fcst_ser)
    tmp_vis.extend(meps_ser)
    #print("Forecast visibility classes:", fcst_data["fcst_vis"].tolist())
    #print("MEPS visibility classes:", dfs["meps_class"].tolist())
    #print("Combined visibility class:", tmp_vis)

    df_out = pd.DataFrame({
        "leadtime": np.arange(len(tmp_vis)),
        "pred_ml": tmp_vis
    })
    """
    # Plot each of the 9 classes
    lead_hours = np.arange(1, fcst_probs.shape[0] + 1)
    plt.figure()
    for cls in range(fcst_probs.shape[1]):
        plt.plot(lead_hours, fcst_probs[:,cls], label=f"class {cls+1}")
    plt.xlabel("Lead time [h]")
    plt.ylabel("Predicted probability")
    plt.title("Visibility‐class probabilities over lead time")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vis_pred_probs.png")
    plt.close()
    """
    return df_out

def preprocess_ceiling(df_data):
    df = df_data.copy()
    #rename columns for clarity
    df = df.rename(columns={
        't10m': 't65', 
        'ppa': 'sp', 
        'vis': 'visibility', 
        'bld': 'mld', 
        'rh10m': 'rh65',
        'snr': 'snow_hourly',
        'cc': 'tcc',
        'rnetsw': 'nswrs_hourly',
        'rnetlw': 'nlwrs_hourly',
    })
    # Clip visibility to 10000
    df["visibility"] = df["visibility"].clip(upper=10000)
    # Set snr first value to 0
    df.loc[df.index[0],'snr'] = 0
    # Calculate t_inv = t65 - t0m
    df["t_inv"] = df["t65"] - df["t0m"]
    # Calculate hourly values for rain
    df['rain_hourly'] = df['rr'].diff().fillna(0)
    df['rain_hourly'] = df['rain_hourly'].clip(lower=0)
    # Calculate trends
    df["t2m_trend"] = df.t2m.diff().fillna(0)
    df["visibility_trend"] = df.visibility.diff().fillna(0)
    df["rh2m_trend"] = df.rh2m.diff().fillna(0)
    df["rh65_trend"] = df.rh65.diff().fillna(0)
    df["t65_trend"] = df.t65.diff().fillna(0)
    df["ceiling_trend"] = df.ceiling.diff().fillna(0)
    # Add month and hour sin/cos
    df = df.assign(month=df.validdate.dt.month)
    df = df.assign(hour=df.validdate.dt.hour)
    df = encode(df, "month", 12)
    df = encode(df, "hour", 24)
    final_features = [
        'leadtime', 't0m', 't2m', 't65', 'rh2m', 'sp',
       'lcc', 'visibility', 'mld', 'tcc', 'ceiling', 'rh65', 't_inv',
       'snow_hourly', 'rain_hourly', 'nlwrs_hourly', 'nswrs_hourly',
       't2m_trend', 'visibility_trend', 'rh2m_trend', 'rh65_trend',
       't65_trend', 'ceiling_trend', 'ws', 'wd', 'month_sin', 'month_cos',
       'hour_sin', 'hour_cos', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 't_1',
       't_2', 't_3', 't_4', 't_5', 't_6', 'td_1', 'td_2', 'td_3', 'td_4',
       'td_5', 'td_6', 'ws_1', 'ws_2', 'ws_3', 'ws_4', 'ws_5', 'ws_6', 'wd_1',
       'wd_2', 'wd_3', 'wd_4', 'wd_5', 'wd_6', 'cc1_1', 'cc1_2', 'cc1_3',
       'cc1_4', 'cc1_5', 'cc1_6', 'cbase1_1', 'cbase1_2', 'cbase1_3',
       'cbase1_4', 'cbase1_5', 'cbase1_6', 'cbase_1', 'cbase_2', 'cbase_3',
       'cbase_4', 'cbase_5', 'cbase_6', 'vis_1', 'vis_2', 'vis_3', 'vis_4',
       'vis_5', 'vis_6'
    ]
    # Select only the final features
    df = df[final_features]
    # Remove leadtimes <= 3
    df = df[df['leadtime'] > 3]
    #Add new metar features
    # Trend features
    params = ["ws", "cbase1", "cbase", "p", "vis", "cc1"]
    for param in params:
        for i in range(2,7):
            df[param + "_trend_" + str(i)] = df[param + "_1"] - df[param + "_" + str(i)]
    # Mean features
    params = ["ws", "cbase1", "cbase"]
    for param in params:
        for i in [2,4,6]:
            # Calculate mean of the parameter from 1 to i
            sum_param = 0
            for j in range(1,i+1):
                sum_param += df[param + "_" + str(j)]
            df[param + "_mean_" + str(i)] = sum_param / i
    # New parameter t-td
    for i in range(1,7):
        df["t_td_" + str(i)] = df["t_" + str(i)] - df["td_" + str(i)]
    return df

def ml_forecast_ceiling(df, args):
    """
    Make a forecast with ML
    """
    # Create empty numpy array of length df
    fcst_probs = []
    for i in range(2):
        if (i == 0):
            model_path = args.model_cbase_4_9 #'xgb_cbase_4_9_20250610.json'
            df_p = df[(df["leadtime"] < 10)]
        else:
            model_path = args.model_cbase_0_36 #'xgb_cbase_0_36_20250610.json'
            df_p = df[(df["leadtime"] >= 10)]
        model = xgb.Booster()
        model.load_model(model_path)
        # Predict using the model
        tmp = model.predict(xgb.DMatrix(df_p))
        fcst_probs.append(tmp)
    # Stack lists
    fcst_probs = np.concatenate(fcst_probs, axis=0)
    # get the forecasted class
    fcst_ceiling = fcst_probs.argmax(axis=1) 
    # Second best if model is unsure about the class 9
    fcst_data = pd.DataFrame(fcst_probs, columns=[f"Class{i}" for i in range(fcst_probs.shape[1])])
    fcst_data["fcst_ceiling"] = fcst_ceiling  # Add predicted class
    # Convert latest observation to ceiling class
    bins = [0, 100, 200, 500, 1000, 1500, np.inf]
    labels = [
        "<=100 m",
        "100-200 m",
        "200-500 m",
        "500-1000 m",
        "1000-1500 m",
        ">1500 m"
    ]
    obs_ceiling = pd.cut(df["cbase_1"], bins=bins, labels=labels, right=True).cat.codes
    obs0 = obs_ceiling.iloc[0]
    tmp_ceiling = [obs0]  # Start with the observation class
    # Extend the list with forecast visibility classes and MEPS visibility classes
    fcst_ser = fcst_data["fcst_ceiling"].tolist()
    tmp_ceiling.extend(fcst_ser)

    df_out = pd.DataFrame({
        "leadtime": np.arange(len(tmp_ceiling)),
        "pred_ml": tmp_ceiling
    })
    return df_out


def main():
    args = parse_command_line()
    # Obs lat/lon
    latlon = pd.DataFrame(   {
            "lat": [60.329368],
            "lon": [24.97274]
        }
    )
    idxs = np.array([
        [418, 643],
        [418, 644],
        [417, 643],
        [417, 644]
    ], dtype=np.int32)
    
    st = time.time()
    analysistime, forecasttime, lons, lats, clb, wd, ws, t2m, t10m, t0m, ppa, vis, rh2m, rh10m, bld, rr, snr, cc, lcc, rnetsw, rnetlw = read_grid(args)
    analysistime = pd.to_datetime(analysistime)
    forecasttime = pd.to_datetime(forecasttime)
    print("Analysistime",analysistime)
    
    # Create gridpp location classes and 4 closest points for ceiling
    grid = gridpp.Grid(lats,lons)
    points = gridpp.Points(latlon["lat"].to_numpy(),latlon["lon"].to_numpy())
    #idxs = grid.get_closest_neighbours(float(latlon.loc[0, "lat"]),float(latlon.loc[0, "lon"]), 4)
    
    # Interpolate all MEPS data to the airport point
    df_data = point_interpolate(analysistime, forecasttime, clb, wd, ws, t2m, t10m, t0m, ppa, vis, rh2m, rh10m, bld, rr, snr, cc, lcc, rnetsw, rnetlw, grid, points, idxs)
    # retrieve obs (METAR)
    icao = 'EFHK'  
    atimestr = analysistime.strftime("%Y%m%d%H%M")
    latest_metar = latest_metar_observations_features(icao, analysistime.strftime("%Y%m%d%H%M"))
    # Combine wiht MEPS data
    s = latest_metar.iloc[0]
    df_data = df_data.assign(**s.to_dict())
    # Preprocess the data for ML
    if args.parameter == "visibility":
        df = preprocess_vis(df_data)
        # Make a forecast with ML
        ml = ml_forecast_vis(df, args)
        # ml is a df with ml forecast visibility (until leadtime 18h) and MEPS visibility (after 18h)
    elif args.parameter == "cldbase":
        df = preprocess_ceiling(df_data)
        # Make a forecast with ML
        ml = ml_forecast_ceiling(df, args)
    # Make a grid with missing data, except for the area around the airport
    new_grid = clb.copy() # could be any of the MEPS data
    new_grid = new_grid[3:]  # Remove first 3 leadtimes (0, 1, and 2 hours)
    # Assign all values as missing data == 9999
    new_grid[:] = 9999 
    # Get grid points near the airport in 20km radius
    gidxs = grid.get_neighbours(float(latlon.loc[0, "lat"]),float(latlon.loc[0, "lon"]), 20000)
    #print("Grid points near the airport:", gidxs)
    # assign the ml forecast to grid points near airport to the new_grid
    for t, val in enumerate(ml["pred_ml"].values):
        for i, j in gidxs:
            new_grid[t, i, j] = val

    # Save data to grib file 
    # Modify the times: 
    new_fcsttime = forecasttime[2:]
    write_grib(args, analysistime, new_fcsttime, new_grid, args.grib_options)
    """        
    # plot the grid for 8 the leadtimes
    # Define the Lambert Conformal projection with appropriate parameters
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    central_lon = (lons.min() + lons.max()) / 2
    central_lat = (lats.min() + lats.max()) / 2
    projection = ccrs.LambertConformal(
    central_longitude=central_lon,
    central_latitude=central_lat,
    standard_parallels=(25, 25)
    )
    for k in range(0, 8):
        fig = plt.figure(figsize=(6, 6), dpi=80)
        ax  = fig.add_subplot(1, 1, 1, projection=projection)

        # set the geographic extent [west, east, south, north]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

        # draw coastlines, borders, etc
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'))
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # now plot your data, telling Cartopy these are lon/lat coords
        mesh = ax.pcolormesh(
            lons, lats, new_grid[k+1],
            transform=ccrs.PlateCarree(),   # important!
            cmap="Spectral_r",
            vmin=0, vmax=9
        )

        cbar = plt.colorbar(
            mesh, ax=ax,
            orientation="horizontal",
            pad=0.05,
            label=f"VIS {k+1}h"
        )

        plt.savefig(f"vis_{k+1}h.png", bbox_inches="tight")
        plt.close()
    """



    """
    print(f"clb min/max:, {np.min(clb):.1f} {np.max(clb):.1f}")
    print(f"wd min/max:, {np.min(wd):.1f} {np.max(wd):.1f}")
    print(f"ws min/max:, {np.min(ws):.1f} {np.max(ws):.1f}")
    print(f"t2m min/max:, {np.min(t2m):.1f} {np.max(t2m):.1f}")
    print(f"t10m min/max:, {np.min(t10m):.1f} {np.max(t10m):.1f}")
    print(f"t0m min/max:, {np.min(t0m):.1f} {np.max(t0m):.1f}")
    print(f"ppa min/max:, {np.min(ppa):.1f} {np.max(ppa):.1f}")
    print(f"vis min/max:, {np.min(vis):.1f} {np.max(vis):.1f}")
    print(f"rh2m min/max:, {np.min(rh2m):.1f} {np.max(rh2m):.1f}")
    print(f"rh10m min/max:, {np.min(rh10m):.1f} {np.max(rh10m):.1f}")
    print(f"bld min/max:, {np.min(bld):.1f} {np.max(bld):.1f}")
    print(f"rr min/max:, {np.min(rr):.1f} {np.max(rr):.1f}")
    print(f"snr min/max:, {np.min(snr):.1f} {np.max(snr):.1f}")
    print(f"cc min/max:, {np.min(cc):.1f} {np.max(cc):.1f}")
    print(f"lcc min/max:, {np.min(lcc):.1f} {np.max(lcc):.1f}")
    print(f"rnetsw min/max:, {np.min(rnetsw):.1f} {np.max(rnetsw):.1f}")
    print(f"rnetlw min/max:, {np.min(rnetlw):.1f} {np.max(rnetlw):.1f}")
    """


if __name__ == "__main__":
    main()