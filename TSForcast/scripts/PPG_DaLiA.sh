#!/bin/bash


for src in cycling drive sitting walking
do
    for trg in cycling drive sitting walking
    do
        if [ $src != $trg ]; then
            python run.py --lr 0.01 --dataset PPG-DaLiA --src $src --trg $trg


        fi
    done
done