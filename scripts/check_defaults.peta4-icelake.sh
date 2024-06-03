#!/bin/bash
#SBATCH -J check_defaults
#SBATCH -A MLMI-gtj21-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=check_defaults.out
#SBATCH --error=check_defaults.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

echo "Job ID: $SLURM_JOB_ID" > check_defaults.out
echo "Allocated Nodes: $SLURM_JOB_NUM_NODES" >> check_defaults.out

# Using scontrol to fetch job details
job_info=$(scontrol show job $SLURM_JOB_ID)

# Extracting number of tasks
num_tasks=$(echo "$job_info" | grep -oP 'NumNodes=\K\d+')

# Extracting CPUs per task
cpus_per_task=$(echo "$job_info" | grep -oP 'NumCPUs=\K\d+')

# Extracting memory per CPU
memory_per_cpu=$(echo "$job_info" | grep -oP 'TRES=cpu=\d+,mem=\K[^,]+')

echo "Allocated Tasks: $num_tasks" >> check_defaults.out
echo "Allocated CPUs per Task: $cpus_per_task" >> check_defaults.out
echo "Allocated Memory per CPU: $memory_per_cpu" >> check_defaults.out
