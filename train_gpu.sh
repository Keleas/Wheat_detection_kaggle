#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=HuaweiV100
#SBATCH --mail-user=grishin.na@phystech.edu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --job-name="Train panda-cancer image classifier"
#SBATCH --comment="Settings and trainning script (cl)"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/AI/grishin.na/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/AI/grishin.na/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/AI/grishin.na/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/AI/grishin.na/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Anaconda
conda activate torch

# Run
python src/train_cl.py \
--model efficientnet-b0 \
--cuda \
--tile_size 256 \
--n_tiles 36 \
--level 1 \
--hyp_params cfg/hyp_train_cl.json \
--use_neptune_log \
--neptune_params cfg/neptune_settings_cl.json \
--results_name results.txt \
> /home/AI/grishin.na/panda/out.txt