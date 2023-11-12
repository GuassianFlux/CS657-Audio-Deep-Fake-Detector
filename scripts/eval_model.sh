#!/bin/bash
count=$1
filename=$2

for i in $(seq 1 $count)
do
    file=
    sh workflows/pv.sh > "$filename"_"$i".txt
done