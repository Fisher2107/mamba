#!/bin/bash

# 1st arg - job name
# 2nd arg - runtime

# Define the job script file name based on the first argument
job_script="jobs/${1}${2}.sh"

# clear file contents
> "$job_script"

# Create the job script file
echo "#!/bin/bash" >> "$job_script"
echo "" >> "$job_script"

echo "# Grid Engine options" >> "$job_script"
echo "#\$ -N ${1}  # Name of the job" >> "$job_script"
echo "#\$ -wd /exports/eddie/scratch/s2517783/mamba # Run the job from the scratch directory" >> "$job_script"
echo "#\$ -l h_rt=${2}:00:00  # Request a runtime" >> "$job_script"
echo "#\$ -q gpu          # Submit the job to the gpu queue" >> "$job_script"
echo "#\$ -pe gpu-a100 1  # Request NNODE A100 GPUs" >> "$job_script"
echo "#\$ -l h_vmem=80G    # Request memory per core" >> "$job_script"
echo "#\$ -l rl9=true    # rocky linux update" >> "$job_script"
echo "" >> "$job_script"

echo "# Load the module system" >> "$job_script"
echo ". /etc/profile.d/modules.sh" >> "$job_script"
echo ". /exports/applications/support/set_qlogin_environment.sh" >> "$job_script"
echo "" >> "$job_script"

echo "# Load the CUDA module" >> "$job_script"
echo "module load cuda/12.1.1" >> "$job_script"
echo "" >> "$job_script"

echo "# Activate the conda environment for CUDA" >> "$job_script"
echo "source /exports/csce/eddie/inf/groups/dawg/miniconda3/bin/activate base" >> "$job_script"

echo "cd /exports/eddie/scratch/s2517783/mamba" >> "$job_script"

echo "conda activate tspp" >> "$job_script"
echo "" >> "$job_script"

# Copy run.sh to jobs directory with new name
echo "cp run.sh jobs/run_${1}.sh" >> "$job_script"

# Add command to run the inner script
echo "bash run.sh" >> "$job_script"