#!/bin/bash

SCRIPT_PATH=$(dirname $(readlink -f $0))
ROOT_DIR=${SCRIPT_PATH%/*}

WRF_DIR=$ROOT_DIR/WRF
WPS_DIR=$ROOT_DIR/WPS
DATA_DIR=/data

source /home/wrfuser/miniconda3/bin/activate wrf

# ========================== Run WPS ==========================
python $ROOT_DIR/pipeline/templates/render_templates.py --wrf_root $ROOT_DIR

cd $WPS_DIR
echo "Running geogrid"
rm -f log.geogrid
rm -f geo_em.d*
./geogrid.exe &> log.geogrid
echo "--- Completed"

python $ROOT_DIR/pipeline/main.py