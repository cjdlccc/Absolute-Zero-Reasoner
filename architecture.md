# Absolute Zero Reasoner：代码架构

本文依据当前仓库代码，概述项目模块、训练流程和模块边界。项目实现面向代码推理的自博弈强化学习训练；分布式 PPO、Ray worker 和模型 rollout 等底层能力主要由外部 veRL 提供，本仓库实现 AZR 的任务构造、执行验证、奖励计算和训练循环扩展。

## 总体流程

    Hydra 配置与启动脚本
            │
            ▼
    main_azr_ppo.py：初始化 Ray、Tokenizer、worker、reward manager
            │
            ▼
    azr_ray_trainer.py：准备种子数据 → PROPOSE → 校验与筛选 → SOLVE → PPO 更新
            │                                  │                 │
            │                                  ▼                 ▼
            │                         Python / SandboxFusion   CodeIORewardManager
            │                                                    │
            └──────────────────── veRL PPO / Ray workers ◀───────┘
                                      │
                                      ▼
                              checkpoint、指标、训练数据

训练按照配置中的 problem_types 循环。code_i、code_o、code_e、code_f 分别对应预测程序输入、预测程序输出、预测错误类型，以及从示例中归纳函数。启用 train_propose 后，模型既生成任务/程序（PROPOSE），也尝试解决生成或采样得到的任务（SOLVE）；训练数据池会随迭代更新。

## 目录与职责

| 路径 | 职责 |
| --- | --- |
| absolute_zero_reasoner/main_azr_ppo.py | Hydra 训练入口。启动 Ray，加载 tokenizer/processor，建立 Ray 资源池与 worker 映射，并创建奖励管理器和自定义 trainer。 |
| absolute_zero_reasoner/configs/azr_ppo_trainer.yaml | 训练、模型、数据、奖励、执行器、任务类型和数据选择策略的默认配置。启动脚本及 Hydra 命令行参数可覆盖配置。 |
| absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py | AZR 主要编排逻辑：种子/任务数据管理、生成与求解批次、执行器生命周期、PPO 批次计算、验证、指标、数据及 checkpoint 保存。复用 veRL PPO trainer 的 worker 更新能力。 |
| absolute_zero_reasoner/trainer/ppo/reason_rl_ray_trainer.py | 基于 veRL 的 PPO trainer 扩展，处理 PPO 配置、dataloader、验证及训练循环基础逻辑，供项目 trainer 复用。 |
| absolute_zero_reasoner/data_construction/ | 将代码片段和 IO 样例转换为生成任务或预测任务，包括 prompt 模板、样例处理、过滤及 Parquet 数据构造。 |
| absolute_zero_reasoner/rewards/ | 解析模型输出、检查生成内容、估计求解准确率并计算生成任务奖励；包含代码复杂度、多样性等奖励组件及数学评测辅助逻辑。 |
| absolute_zero_reasoner/utils/code_utils/ | 代码解析、模板、静态检查，以及 Python 和 SandboxFusion 执行器适配。 |
| absolute_zero_reasoner/utils/dataset/rl_dataset.py | 将 Parquet 记录加载为 veRL rollout 使用的 RL 数据集。 |
| absolute_zero_reasoner/utils/ | 跟踪与日志、辅助函数、checkpoint 转换、tokenizer 修正等工具。 |
| scripts/seeding/ | 各模型的种子数据准备脚本。 |
| scripts/selfplay/ | 各模型的自博弈训练启动脚本，通常通过 Hydra 覆盖模型和训练参数。 |
| evaluation/ | 数学和代码基准评测入口及相关工具/数据，独立于训练运行时核心模块。 |
| data/ | 预生成或导出的 seed IO JSONL 数据。训练过程中也可能在运行目录中动态构造并保存数据。 |

## 训练组件交互

1. main_azr_ppo.py 读取 Hydra 配置并启动 Ray TaskRunner。根据配置选择 FSDP/FSDP2 或 Megatron worker 组，将 Actor/Rollout、Critic 以及可选的 RefPolicy/RewardModel 绑定到 Ray 资源池。
2. Runner 加载 tokenizer 和 processor，创建训练及验证使用的 CodeIORewardManager，将规则奖励和验证奖励接入 veRL trainer。
3. CodeIORayPPOTrainer 准备已有数据，或调用数据构造逻辑补足 seed 数据。任务提示词由 data_construction/prompts.py 提供，constructor.py 将其组装为 rollout 可用的记录。
4. Trainer 针对配置的任务类型生成 PROPOSE 样本。输出经过解析、格式/安全检查后交给执行器运行；无效或不符合配置的数据按策略过滤，有效样本进入任务/种子池。
5. SOLVE 阶段将题目送到 rollout worker 采样回答。CodeIORewardManager 解析答案并调用执行器验证，例如运行程序核对预测输入/输出，或使用隐藏样例检查函数。生成任务还可按准确率、复杂度、编辑距离和答案类型多样性等配置组合奖励。
6. veRL PPO trainer 使用 rollout、奖励、Critic/参考策略等信息计算并执行策略更新。Trainer 记录指标、更新数据池，并按配置运行验证和保存 checkpoint。

具体启用的任务、是否训练 PROPOSE、执行器类型和奖励组合，以 azr_ppo_trainer.yaml 与实际启动命令为准。执行器可使用本地 Python 或 SandboxFusion。由于它会运行模型生成的代码，部署时应使用隔离环境。

## 关键接口与数据形态

- 数据构造输出使用统一记录字段，例如 prompt（chat 消息）、data_source、problem、reward_model 和 extra_info。extra_info 携带 split、索引、任务 metric、参考样例等训练或评分上下文。
- rollout 后，reward manager 从 DataProto 读取输入和模型响应，按任务类型解码结构化答案并计算准确率、生成质量等奖励，供 trainer 转为 PPO 所需的批次奖励。
- 生成代码、参考程序、输入输出和执行器结果在 trainer/reward 层之间传递。Parquet 用于训练 dataset 的临时或持久化交接，JSONL 用于 seed 数据交换。
- 训练输出目录由配置的输出根路径、训练数据名、模型名及答案提取类型组合得到。实际 checkpoint、日志和数据文件名以当前配置及 veRL 版本为准。

## 常见改动位置

- 增加或修改任务提示词：absolute_zero_reasoner/data_construction/prompts.py。
- 修改生成数据形态、采样或样例转换：data_construction/constructor.py、process_data.py。
- 调整自博弈编排、数据池或训练阶段：trainer/ppo/azr_ray_trainer.py。
- 修改输出抽取、代码有效性或奖励计算：rewards/code_reward.py、rewards/reward_managers.py。
- 增加执行器或改变代码验证方式：utils/code_utils/，并检查 trainer 对执行器生命周期的调用。
- 调整训练选项和默认值：configs/azr_ppo_trainer.yaml 及对应 scripts/ 启动脚本。
- 修改外部 veRL 行为时，同时核查依赖版本及本仓库 trainer 的继承关系和接口使用。Ray worker、PPO 优化器等实现并不完全由本仓库维护。

## 架构边界

本仓库自有代码主要负责 AZR 的任务生成和数据筛选策略、代码执行验证、奖励设计、训练循环扩展与实验配置。分布式调度、Actor/Rollout/Critic worker 的核心实现、PPO 优化细节和模型后端主要依赖 veRL、Ray、Transformers 等库。evaluation/ 下的 benchmark 工具独立于训练入口，其依赖和数据集按各子目录说明管理。
