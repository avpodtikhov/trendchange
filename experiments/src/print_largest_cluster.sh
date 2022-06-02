#!/bin/bash
#SBATCH -c 1

source /home/aarukavishnikov/apodtikhov/env_default/bin/activate
python print_largest_cluster.py