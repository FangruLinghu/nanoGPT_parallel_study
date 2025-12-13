source /pscratch/sd/e/es_lh/venv/dist_dl_hw/bin/activate

cd /pscratch/sd/e/es_lh/nanoGPT/MP

# 2 gpu
# python train_mp.py config/train_shakespeare_char.py --compile=False 2>&1 | tee output_2g.log

# 4 gpus (1 node)
python train_mp_logger.py /pscratch/sd/e/es_lh/nanoGPT/config/train_shakespeare_char.py --compile=False 2>&1 | tee output_4g_xxl.log

# 8 gpus (2 nodes)
# srun -N 2 -n 2 --ntasks-per-node=1 --gpus-per-task=4 bash -c '\
# MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1) NODE_RANK=${SLURM_PROCID}
# MASTER_PORT=29500
# echo "WORLD_SIZE" $WORLD_SIZE "NODE_RANK" $NODE_RANK "MASTER_PORT" $MASTER_PORT
# echo "Rank: $SLURM_PROCID, Local Rank: $SLURM_LOCALID, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# torchrun \
# 	--nnodes=2 \
# 	--nproc_per_node=4 \
# 	--node_rank=$SLURM_PROCID \
# 	--master_addr=$MASTER_ADDR \
# 	--master_port=$MASTER_PORT \
# rpc_train_v3.py 2>&1 | tee output_2n.log
# '

