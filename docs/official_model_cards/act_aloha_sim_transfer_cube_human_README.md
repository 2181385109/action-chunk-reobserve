---
license: apache-2.0
datasets:
- lerobot/aloha_sim_transfer_cube_human
tags:
- act-policy
- model_hub_mixin
- pytorch_model_hub_mixin
- robotics
pipeline_tag: robotics
---
# Model Card for ACT/AlohaTransferCube

Action Chunking Transformer Policy (as per [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)) trained for the `AlohaTransferCube` environment from [gym-aloha](https://github.com/huggingface/gym-aloha).

![demo](demo.gif)

## How to Get Started with the Model

See the [LeRobot library](https://github.com/huggingface/lerobot) (particularly the [evaluation script](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py)) for instructions on how to load and evaluate this model.

## Training Details

Trained with [LeRobot@3c0a209](https://github.com/huggingface/lerobot/tree/3c0a209f9fac4d2a57617e686a7f2a2309144ba2).

The model was trained using [LeRobot's training script](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/train.py) and with the [aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) dataset, using this command:

```bash
python lerobot/scripts/train.py \
    --output_dir=outputs/train/act_aloha_transfer \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --wandb.enable=true
```

The training curves may be found at https://wandb.ai/aliberts/lerobot/runs/720l37xb.
The current model corresponds to the checkpoint at 80k steps.

This took about 1h45 to train on an Nvida A100.

## Evaluation

The model was evaluated on the `AlohaTransferCube` task from [gym-aloha](https://github.com/huggingface/gym-aloha) and compared to a similar model trained with the original [ACT repository](https://github.com/tonyzhaozh/act). Each episode marks a success if the cube is successfully picked by one robot arm and transferred to the other robot arm.

Here are the success rate results for 500 episodes worth of evaluation. The "Theirs" column is for an equivalent model trained on the original ACT repository and evaluated on LeRobot (the model weights may be found in the [`original_act_repo`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/original_act_repo) branch of this respository). The results of each of the individual rollouts may be found in [eval_info.json](eval_info.json).

<blank>|Ours|Theirs
-|-|-
Success rate for 500 episodes (%) | 83.0 | 68.0

It was produced after training with this command:
```bash
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/act_aloha_transfer/checkpoints/080000/pretrained_model \
    --output_dir=outputs/eval/act_aloha_transfer/080000 \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --device=cuda \
    --use_amp=false
```

The original code was heavily refactored, and some bugs were spotted along the way. The differences in code may account for the difference in success rate. Another possibility is that our simulation environment may use slightly different heuristics to evaluate success (we've observed that success is registered as soon as the second arm's gripper makes antipodal contact with the cube). Finally, one should observe that the in-training evaluation jumps up towards the end of training. This may need further investigation (Is it statistically significant? If so, what is the cause?).
