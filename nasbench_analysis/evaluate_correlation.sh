#!/bin/bash
#SBATCH -p meta_gpu-ti  # partition (queue)
#SBATCH --mem 30000 # memory pool for all cores
#SBATCH -t 11-00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -a 1-10 # array size
#SBATCH --gres=gpu:1  # reserves four GPUs
#SBATCH -D /home/siemsj/projects/darts/cnn # Change working_dir
#SBATCH -o log/log_$USER_%Y-%m-%d.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/err_$USER_%Y-%m-%d.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J correlation # sets the job name. If not specified, the file name will be used as job name
# #SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate conda environment
source ~/.bashrc
conda activate pytorch1_0_1

gpu_counter=1

for epoch in 0 10 20 30 40 50 60 70 80 90
    do
        # Job to perform
        if [ $gpu_counter -eq $SLURM_ARRAY_TASK_ID ]; then
           PYTHONPATH=$PWD python nasbench_analysis/evaluate_one_shot_models.py --epoch=${epoch} --model_path=/home/siemsj/projects/darts/cnn/experiments/random_ws/ss_20190905-111250_3_0
           exit $?
        fi

        let gpu_counter+=1
done
# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";