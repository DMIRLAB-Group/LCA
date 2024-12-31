#!/bin/bash


for src in greeting eating walking smoking
do
    for trg in greeting eating walking smoking
    do
        if [ $src != $trg ]; then
            python run.py --lr 0.005 --dataset human_action --src $src --trg $trg

        fi
    done
done