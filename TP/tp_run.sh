source /pscratch/sd/e/es_lh/venv/dist_dl_hw/bin/activate

cd /pscratch/sd/e/es_lh/nanoGPT/TP

# 4 gpus (1 node)
torchrun --standalone --nproc_per_node=4 train_tp_logger.py /pscratch/sd/e/es_lh/nanoGPT/config/train_shakespeare_char.py 2>&1 | tee output_4g_l.log

# 8 gpus (2 nodes)
# srun -N 2 -n 2 --ntasks-per-node=1 --gpus-per-task=4 bash -c '\
# MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1) NODE_RANK=${SLURM_PROCID}
# MASTER_PORT=29501
# echo "WORLD_SIZE" $WORLD_SIZE "NODE_RANK" $NODE_RANK "MASTER_PORT" $MASTER_PORT
# echo "Rank: $SLURM_PROCID, Local Rank: $SLURM_LOCALID, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# torchrun \
# 	--nnodes=2 \
# 	--nproc_per_node=4 \
# 	--node_rank=$SLURM_PROCID \
# 	--master_addr=$MASTER_ADDR \
# 	--master_port=$MASTER_PORT \
# train_tp_v2.py config/train_shakespeare_char.py 2>&1 | tee output_8g.log
# '

