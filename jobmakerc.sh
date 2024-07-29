#!/bin/bash

# 1st arg - runtime in hours of each job
# 2nd arg - job name

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
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=${1}:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=tc064-s2517783

# Load the required modules
module load nvidia/nvhpc/24.5
pwd
source ../miniconda3/bin/activate
conda activate base

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