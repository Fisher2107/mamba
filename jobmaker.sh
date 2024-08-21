#!/bin/bash

# 1st arg - runtime in hours of each job
# 2nd arg - job name
# 3rf arg - memory in GB

# Counter for job number
job_count=1

# Read run.sh line by line
while IFS= read -r line
do
    # Check if the line starts with "python"
    if [[ $line == python* ]]
    then
        # Create a new job script for each Python command
        job_file="jobs/job_${2}_${job_count}.sh"
        
        # Write the job script
        cat << EOF > "$job_file"
#!/bin/bash
#$ -N job_${2}_${job_count}
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=${1}:00:00
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=${3}G
#$ -l rl9=true

. /etc/profile.d/modules.sh
. /exports/applications/support/set_qlogin_environment.sh

module load cuda/12.1.1

source /exports/eddie/scratch/s2517783/miniconda3/bin/activate base
cd /exports/eddie/scratch/s2517783/mamba
conda activate tsp

$line
EOF
        # Make the job script executable
        chmod +x "$job_file"
        # Increment job count
        ((job_count++))
    fi
done < run.sh

((job_count--))
echo "Created ${job_count} jobs."
