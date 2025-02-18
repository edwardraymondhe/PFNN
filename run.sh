#!/bin/bash
cd ./demo
make
cd ../

rm -rf ./demo/network/pfnn
mkdir ./demo/network/pfnn
python preprocess_footstep_phase.py
python generate_patches.py
python generate_database.py
python train_pfnn.py
