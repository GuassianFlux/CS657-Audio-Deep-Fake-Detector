#!/bin/bash
count=$1
filename=$2

for i in $(seq 1 $count)
do
    file=
    python src > "$filename"_"$i".txt
done