#!/bin/bash
# 在全新 WSL shell 里验证环境（完成标准 2）
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
python -c "import lerobot, torch; print('lerobot', lerobot.__version__ if hasattr(lerobot,'__version__') else __import__('importlib.metadata').metadata.version('lerobot'), '| torch', torch.__version__, '| cuda_available', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0))"
echo "MUJOCO_GL=$MUJOCO_GL CHUNK_RENDER=$CHUNK_RENDER GALLIUM_DRIVER=$GALLIUM_DRIVER HF_ENDPOINT=$HF_ENDPOINT"
