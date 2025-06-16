# Machine learning based (XGB) model for visibility and cloud base height class forecasting
This kairos inference code can be used to predict visibility and cloud base height (cldbase) class forecasts for 3-36 hour leadtimes. The input data is the MEPS model grids (grib2) and the output is class forecast model fields (grib2). Currently the forecasts are produced only around the Helsinki-Vantaa airport (EFHK). Machine learning model is an eXtreme Gradient Boosting (XGBoost) multi-class classification model which is based on 5 years of training data (training code is not included in this repo). 

## Usage
Running with run_inference.sh shell script:
```
./run_inference.py YYYYMMMDDHH parameter output.grib2 producer_id
```
E.g.
```
./run_inference.py 2025061209 visibility visibility_class.grib2 203
```

## Authors
kaisa.ylinen@fmi.fi
leila.hieta@fmi.fi