#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=nvr_elm_llm
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --output=./log/%A_%x.out
#SBATCH --error=./log/%A_%x.err
#SBATCH --job-name=nvr_elm_llm:sgl-deploy
#SBATCH --array=1-60%1

set -e
model_path=$1
captions=$2
output_folder=$3

# IMPORTANT: workaround OUTLINES
#   sglang will replace outlines with xgrammar in the future.
export OUTLINES_CACHE_DIR="/tmp/cache"

# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Running on nodes: [$nodes]"
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))


# __doc_head_ray_start__
port=8265
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --gpus-per-task="${SLURM_GPUS_PER_TASK}" \
    python3 -m sglang.launch_server --model-path $model_path \
        --tp 32 --dist-init-addr $head_node_ip:5000 --nnodes $SLURM_JOB_NUM_NODES --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000 &

# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        --gpus-per-task="${SLURM_GPUS_PER_TASK}" \
        python3 -m sglang.launch_server --model-path $model_path \
            --tp 32 --dist-init-addr $head_node_ip:5000 --nnodes $SLURM_JOB_NUM_NODES --node-rank $i --trust-remote-code &
    sleep 5
done
# __doc_worker_ray_end__


# for deepseek-v3/r1, it usually take ~30 mins to load
while ! nc -z localhost 30000; do
    sleep 30
    echo "[INFO] Waiting for localhost:30000 to accept connections"
done

echo "[INFO] localhost:30000 is ready to accept connections"

# __doc_script_start__
echo "SGLang init finished. Starting the script"
# sleep infinity

##################################################################################################
# the actual run command 
# this will be runned at the headnode by default
python step4_gen_reasoning_data.py --model_path $model_path --captions $captions --output_folder $output_folder
sleep infinity
