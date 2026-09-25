# Absolute Zero Capability Shift Reward 实验说明

## vLLM 与模型接口

本项目默认使用 vLLM，但 vLLM 由 veRL/Ray 在训练进程内部管理，不需要先启动独立的 vllm serve HTTP 服务。模型不是通过 OpenAI API client 调用。

配置文件：absolute_zero_reasoner/configs/azr_ppo_trainer.yaml

```yaml
actor_rollout_ref:
  model:
    path: ~/models/deepseek-llm-7b-chat
  rollout:
    name: vllm
    mode: sync
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.85
    max_num_batched_tokens: 8192
```

训练代码调用的模型接口：

```python
self.actor_rollout_wg.generate_sequences(batch)
self.actor_rollout_wg.compute_log_prob(batch)
self.ref_policy_wg.compute_ref_log_prob(batch)
```

对应代码主要在 absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py 的 _compute_batch()。

## 训练入口与流程

- 训练入口：absolute_zero_reasoner/main_azr_ppo.py
- proposer/solver rollout：absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py
- verifier/reward：absolute_zero_reasoner/rewards/reward_managers.py
- 默认配置：absolute_zero_reasoner/configs/azr_ppo_trainer.yaml
- 自博弈脚本：scripts/selfplay/
- RL：veRL Ray PPO，默认 algorithm.adv_estimator=gae

gen_code_* 是 proposer 任务，pred_code_* 是 solver 任务。trajectory 使用 veRL DataProto，关键字段为 prompts、responses、response_mask、old_log_probs、token_level_scores 和 token_level_rewards。

## 训练环境

```bash
cd D:/proj/Absolute-Zero-Reasoner
conda env create -f azr_env.yml
conda activate azr
pip install -r requirements.txt
```

需要 CUDA、PyTorch、Ray、veRL 和 vLLM 环境。conda 环境名以本机实际配置为准。

## 训练命令

## 启动前需要补齐的配置

vLLM 对大多数 Hugging Face Transformers causal language model 通用，但不是任意模型都能直接使用。模型需要提供可加载的 Transformers 配置、tokenizer 和权重，并与当前 vLLM/Transformers 版本兼容。建议优先使用项目脚本中已经验证过的 Qwen 系列。

至少需要确认：

1. 模型地址：设置 actor_rollout_ref.model.path，可以是 Hugging Face ID 或本地模型目录。

```text
actor_rollout_ref.model.path=Qwen/Qwen2.5-7B
actor_rollout_ref.model.path=/data/models/Qwen2.5-7B
```

本地目录应包含 config.json、tokenizer 文件和模型权重；使用 Hugging Face ID 时需要联网下载或提前完成缓存。

2. 训练和验证数据：selfplay 脚本中的 data/code_reason/test_answer.parquet 只是示例路径，必须确认存在，或覆盖为：

```text
data.train_files=/data/azr/train.parquet
data.val_files=/data/azr/val.parquet
```

3. GPU 和并行度：检查 trainer.n_gpus_per_node、trainer.nnodes、actor_rollout_ref.rollout.tensor_model_parallel_size 和 actor_rollout_ref.actor.ulysses_sequence_parallel_size。tensor parallel size 不应大于可用 GPU 数量。

4. batch 和长度：根据显存调整 data.train_batch_size、actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu、data.max_prompt_length、data.max_response_length 和 actor_rollout_ref.rollout.max_num_batched_tokens。显存不足时优先降低这些值或 gpu_memory_utilization。

5. seed 数据路径：确认以下路径可读写：

```text
azr.seed_dataset
azr.error_seed_dataset
azr.code_f_seed_dataset
azr.output_seed_path
azr.output_error_seed_path
azr.output_code_f_seed_path
```

6. 代码执行器：默认脚本使用 azr.executor=qwq。若使用 SandboxFusion，需要 Docker 和对应镜像，并设置 azr.executor=sandboxfusion。

7. Reference Policy：当 azr.reward.shift_coef>0 时，程序会额外创建冻结 Reference Policy，默认使用同一个初始模型路径。需要为 reference log probability 预留显存：

```text
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64
azr.reward.shift_coef=0.1
```

shift_coef=0.0 时不会启用 Reference Policy。

8. 日志和 checkpoint：使用 W&B 时配置登录状态、trainer.logger、trainer.project_name 和 trainer.experiment_name；同时确保 trainer.default_local_dir 所在磁盘空间足够。

直接使用 Hydra 启动：

```bash
cd D:/proj/Absolute-Zero-Reasoner
python -m absolute_zero_reasoner.main_azr_ppo \
  actor_rollout_ref.model.path=/path/to/your/model \
  actor_rollout_ref.rollout.name=vllm \
  azr.reward.shift_coef=0.0
```

PowerShell 一行示例：

```powershell
python -m absolute_zero_reasoner.main_azr_ppo actor_rollout_ref.model.path=D:/models/Qwen2.5-7B-Instruct actor_rollout_ref.rollout.name=vllm azr.reward.shift_coef=0.0
```

使用已有 selfplay 脚本：

```bash
bash scripts/selfplay/14b.sh
bash scripts/selfplay/14b.sh azr.reward.shift_coef=0.1
```

常用 vLLM 参数：

```text
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.85
actor_rollout_ref.rollout.max_num_batched_tokens=8192
actor_rollout_ref.rollout.max_num_seqs=1024
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.free_cache_engine=False
```

显存不足时优先降低 gpu_memory_utilization 或 max_num_batched_tokens，并确保 tensor parallel size 与 GPU 数量匹配。

## Capability shift reward

```text
R_shift = sum_t [log pi_current(y_t|x,y_<t) - log pi_base(y_t|x,y_<t)]
R_total = R_success + shift_coef * R_shift
```

shift_coef=0.0 时完全复现原始 Absolute Zero：不创建 Reference Policy、不读取 base log probability，原有 reward 和 PPO 流程不变。

shift_coef>0 时，训练入口创建一次冻结 Reference Policy worker，通过 compute_ref_log_prob 得到 base log probability；对 response token 求和 current/base 差值，并加到 response 最后一个有效 token 的 reward。

## 验证

```bash
python -m compileall -q absolute_zero_reasoner
```

lambda=0：reward 与原始实现一致。
lambda>0：应看到 reference log probability 和 reward/shift_mean，且 reward 增量为 shift_coef * sum(old_log_probs - ref_log_prob)。

## 修改文件

- absolute_zero_reasoner/configs/azr_ppo_trainer.yaml
- absolute_zero_reasoner/main_azr_ppo.py
- absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py
- capability_shift_reward.md
