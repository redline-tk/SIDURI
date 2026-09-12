#!/bin/bash
set -e
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "SIDURI Dataset Downloader"
echo "============================================"

echo ""
echo "[1/6] Server Machine Dataset (SMD)"
if [ ! -d "$DATA_DIR/smd/train" ]; then
    git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git /tmp/smd_repo
    mkdir -p "$DATA_DIR/smd"
    cp -r /tmp/smd_repo/ServerMachineDataset/* "$DATA_DIR/smd/"
    rm -rf /tmp/smd_repo
    echo "  -> SMD downloaded to $DATA_DIR/smd/"
else
    echo "  -> SMD already exists, skipping."
fi

echo ""
echo "[2-3/6] SMAP and MSL Spacecraft Telemetry"
echo "  -> https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
echo "  -> kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p $DATA_DIR/smap_msl --unzip"
echo "  -> expected: $DATA_DIR/smap_msl/{train,test}/*.npy and labeled_anomalies.csv"

echo ""
echo "[4/6] BATADAL SCADA Dataset"
echo "  -> https://www.batadal.net"
echo "  -> place dataset03.csv and dataset04.csv into $DATA_DIR/batadal/"

echo ""
echo "[5/6] CICIDS2017"
echo "  -> https://www.unb.ca/cic/datasets/ids-2017.html"
echo "  -> place labeled CSV flow files into $DATA_DIR/cicids2017/"

echo ""
echo "[6/6] UNSW-NB15"
echo "  -> https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo "  -> place CSV files into $DATA_DIR/unsw_nb15/"

echo ""
echo "============================================"
echo "Automated: SMD. Manual: SMAP/MSL, BATADAL, CICIDS2017, UNSW-NB15."
echo "Next: run the preprocessing scripts in scripts/preprocessing/"
echo "============================================"
