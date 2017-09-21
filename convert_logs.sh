#!/bin/bash

MODELS_PATH=models

for log in `ls $MODELS_PATH/*.log`; do
    name=${log#$MODELS_PATH}
    name=${name#/}
    name=${name%.log}
    python plot_log.py $log plots/$name.csv
done
