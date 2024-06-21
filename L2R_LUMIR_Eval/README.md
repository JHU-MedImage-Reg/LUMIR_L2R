# grand-challenge evaluation docker

This directory contains our sample docker image for Grand-challenge evaluation. Please see https://grand-challenge.org/documentation/automated-evaluation/ for more details. 


# L2R24 LUMIR Evaluation

The source code for the evaluation container for Learn2Reg 2024 LUMIR Challenge

Command to run evaluation:\
`python -m evaluation -c1 ground-truth/LUMIR_VAL_Landmark_evaluation_config.json -c2 ground-truth/LUMIR_VAL_Segmentation_evaluation_config.json -i /input_dir/ -d /ground-truth/ -o output/metrics.json`

# Acknowledgement
A large portion of the evaluation code was borrowed from https://github.com/MDL-UzL/L2R
