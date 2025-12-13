source /pscratch/sd/e/es_lh/venv/dist_dl_hw/bin/activate

cd /pscratch/sd/e/es_lh/nanoGPT/Deepspeed

srun hostname | sort | uniq > hostfile
sed -i 's/$/ slots=4/' hostfile
cat hostfile

deepspeed --hostfile hostfile \
          train_ds_logger.py --wandb_log True 2>&1 | tee output_4g_zero1_xxl.log