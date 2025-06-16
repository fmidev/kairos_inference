#!/bin/bash
# Script to run ML realtime forecasts for testing
# Example: ./run_inference.sh 2025061306 visibility 293 
# ./run_inference.sh 2025061306 cldbase 293

#python3.11 --version

#NN_prev=$(date -u -d "@$(( $(date -u +%s) / 10800 * 10800 ))" +%Y%m%d%H)
#NN=$(date -u -d "${NN_prev:0:8} ${NN_prev:8:2}:00:00 UTC -3 hours" +%Y%m%d%H)
NN=$1
VARIABLE=$2
PRODUCER_ID=$3
OUTPUT_FILE=$VARIABLE"_class_"$NN".grib2"

#WEEKDAY=`date +"%a"`
#HOD=`date +"%H"`
#AIKA1=`date "+%Y%m%d%H"  -u`
#HH=2
#NN=$(($AIKA1-$HH)
#echo $NN_prev
bucket="s3://routines-data-prod/aerodrome/preop/"

echo "ANALYSIS_TIME:" $NN
echo "PARAMETER:" $PARAMETER
echo "OUTPUT_FILE:" $OUTPUT_FILE

python3 inference.py \
  --parameter $VARIABLE \
  --clb "${bucket}${NN}00/CLDBASE-M_0.grib2" \
  --wd  "${bucket}${NN}00/DD-D_10.grib2" \
  --ws  "${bucket}${NN}00/FF-MS_10.grib2" \
  --t2m "${bucket}${NN}00/T-K_2.grib2" \
  --t10m "${bucket}${NN}00/T-K_65.grib2" \
  --t0m "${bucket}${NN}00/T-K_0.grib2" \
  --ppa "${bucket}${NN}00/P-PA_0.grib2" \
  --vis "${bucket}${NN}00/VV-M_0.grib2" \
  --rh2m "${bucket}${NN}00/RH-0TO1_2.grib2" \
  --rh10m "${bucket}${NN}00/RH-0TO1_65.grib2" \
  --bld "${bucket}${NN}00/MIXHGT-M_0.grib2" \
  --rr  "${bucket}${NN}00/RACC-KGM2_0.grib2" \
  --snr "${bucket}${NN}00/SNR-KGM2_0.grib2" \
  --cc  "${bucket}${NN}00/N-0TO1_0.grib2" \
  --lcc "${bucket}${NN}00/NL-0TO1_0.grib2" \
  --rnetlw "${bucket}${NN}00/RNETLW-WM2_0.grib2" \
  --rnetsw "${bucket}${NN}00/RNETSW-WM2_0.grib2" \
  --model_vis_0 "xgb_vis_0_$VIS_TAG.json" \
  --model_vis_1 "xgb_vis_1_$VIS_TAG.json" \
  --model_vis_2 "xgb_vis_2_$VIS_TAG.json" \
  --model_cbase_0_36 "xgb_cbase_0_36_$CLDBASE_TAG.json" \
  --model_cbase_4_9 "xgb_cbase_4_9_$CLDBASE_TAG.json" \
  --output $OUTPUT_FILE \
  --producer_id $PRODUCER_ID \
# with plot option
#python3 biasc.py --topography_data "$bucket""$NN"00/Z-M2S2.grib2 --landseacover "$bucket""$NN"00/LC-0TO1.grib2 --t2_data "$bucket""$NN"00/T-K.grib2 --wg_data "$bucket""$NN"00/FFG-MS.grib2 --nl_data "$bucket""$NN"00/NL-0TO1.grib2 --ppa_data "$bucket""$NN"00/P-PA.grib2 --wd_data "$bucket""$NN"00/DD-D.grib2 --q2_data "$bucket""$NN"00/Q-KGKG.grib2 --ws_data "$bucket""$NN"00/FF-MS.grib2 --rh_data "$bucket""$NN"00/RH-0TO1.grib2 --output testi_"$parameter".grib2 --parameter "$pyparam" --plot
