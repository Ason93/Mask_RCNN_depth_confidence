#!/bin/bash
s1="/home/ason/datasets/cityscapes/gtFine/train/aachen/aachen_000000_"
s2="_gtFine_polygons.json"
for((i=1;i<901;i++))
do 
s3=${%06d}
labelme_json_to_dataset ${s1}${s3}${s2}
done
