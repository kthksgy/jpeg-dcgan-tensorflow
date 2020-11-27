#!/bin/bash
b="1000"
e="40"
dataset="mnist"
# command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --dir-name Grayscale"
# echo $command && $command

q="100"
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --dir-name JPEG${q}_NOROUND"
echo $command && $command
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --low-pass-ratio 0.5 --dir-name JPEG${q}_LPF50_NOROUND"
echo $command && $command

q="75"
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --dir-name JPEG${q}_NOROUND"
echo $command && $command
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --low-pass-ratio 0.5 --dir-name JPEG${q}_LPF50_NOROUND"
echo $command && $command

q="50"
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --dir-name JPEG${q}_NOROUND"
echo $command && $command
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --low-pass-ratio 0.5 --dir-name JPEG${q}_LPF50_NOROUND"
echo $command && $command

dataset="fashion_mnist"
# command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --dir-name Grayscale"
# echo $command && $command

q="100"
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --dir-name JPEG${q}_NOROUND"
echo $command && $command
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --low-pass-ratio 0.5 --dir-name JPEG${q}_LPF50_NOROUND"
echo $command && $command

q="75"
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --dir-name JPEG${q}_NOROUND"
echo $command && $command
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --low-pass-ratio 0.5 --dir-name JPEG${q}_LPF50_NOROUND"
echo $command && $command

q="50"
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --dir-name JPEG${q}_NOROUND"
echo $command && $command
command="python main.py -b ${b} -e ${e} --fid --test --preload --dataset ${dataset} --form jpeg --jpeg-quality ${q} --low-pass-ratio 0.5 --dir-name JPEG${q}_LPF50_NOROUND"
echo $command && $command