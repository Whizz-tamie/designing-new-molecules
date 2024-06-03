#!/bin/bash

# Load necessary modules (if needed)
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

# Activate the virtual environment
source /home/${USER}/.bashrc
source '/rds/user/gtj21/hpc-work/designing-new-molecules/pytorch-env/bin/activate'

#! Full path to application executable: 
application="python3 -u /rds/user/gtj21/hpc-work/designing-new-molecules/src/training/run_random_molsearch.py"

# Run options for the application
options=""

# Work directory
workdir=$(pwd)
cd $workdir

# Execute the command
CMD="$application $options"
echo "Running command: $CMD"
eval $CMD
