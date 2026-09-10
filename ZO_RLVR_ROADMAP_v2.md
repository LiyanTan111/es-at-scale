# ZO-RLVR 思路整理 v2(2026-07-01)

## 0. 一句话

不再用 ZO 去差分 0/1 的 reward(v1,已被实验和数学分析证伪),而是保留 GRPO 的 rollout 和组内 advantage,**只把反传那一步换成 GRZO 对连续代理损失 L(θ) = −Σ Â·log π_θ(y|x) 的两点估计**。全程 forward-only、推理级内存、无 critic / ref model / TIS。

---

## 1. 原始动机(不变)

RL 目标 J(θ) = E_x E_{y∼π_θ(·|x)}[R(x,y)]。GRPO 用反传估计 ∇J,内存远超推理。我们想要 forward-only 的估计器,卖点:(i) 推理级内存,可在纯推理硬件/超大模型上做 RLVR;(ii) 无 importance sampling 修正;(iii)(可选扩展)参数空间的轨迹级探索。

## 2. v1 设计与实验事实

v1(见 start.md):per-example 独立扰动 + greedy 解码 + 原始 0/1 reward + 组内归一化,直接用 reward 做 ZO 信号。

Go/no-go 结果(Qwen2.5, Countdown):

- 1.5B shaped G8(iter 228):answer_acc 0.0185 → 0.016(不涨反降),dead_groups 升至 0.75–0.88;shaped 分数上涨全部来自 format 套利。
- 0.5B shaped G8:地板(0.0005 → 0.0015);dead_groups 0.12 是 format 方差造成的假活组。
- 1.5B binary G32:治疗组,结果待定;即使 dead rate 降到 ~0.5 也可能不够(prompt 可解性双峰分布,加大 G 对"根本不会的题"无效)。

## 3. 诊断:v1 的失败是数学本性,不是工程问题

- greedy + raw-reward ZO 优化的不是 J,而是 J_greedy(θ) = E_x[R(greedy_θ(x))]:**分段常数函数**,梯度几乎处处为零,全部信号挤在输出翻转的决策边界上,σ 平滑是唯一的可微性来源。dead groups 是这个目标函数的必然表现。
- "每份噪声只吃一题"(m=1)是对稀疏 binary reward 最脆弱的极端。ES-at-Scale 能work是因为 m=200 的平均把信号变成准连续(代价:无 per-prompt 信用分配、每步 6000 次生成)。
- GRPO 在同样条件下(1.5B, G8, 0.7%→13.9%)能学,因为 (i) T=1 采样天然制造组内多样性,(ii) 活组里反传拿到的是精确梯度。ZO 在这两点上都吃亏,必须换信号来源。

## 4. 核心转向(Angle B):GRZO 差分 GRPO 的代理损失

关键观察:**forward-only ≠ reward-only**。一次 teacher-forcing 前向就能读出 log π_θ(y|x),无需反传。GRPO 的更新本质是对连续损失

L(θ) = −(1/|D|) Σ_{i,g} Â_{i,g} · log π_θ(y_{i,g} | x_i)

做梯度下降,且该损失在采样点处的梯度**精确等于策略梯度**。于是:

### 每步流程

1. **Rollout(与 GRPO 完全相同)**:每个 prompt x_i 从 π_θ 采样 G 个回答(T=1,vLLM);0/1 判分;组内归一化得 Â_{i,g}。可选 DAPO 式动态采样:全同分的组丢弃补采。
2. **构造代理损失**:D = {(x_i, y_{i,g}, Â_{i,g})},L 如上。
3. **GRZO 两点估计替代反传**:每条 (x_i, y_{i,g}) 一份独立扰动(seed 重生成、in-place、Flipout 式共享基底);前向算 ℓ± = −Â·log π_{θ±σΔW}(y|x);δ = ℓ⁺ − ℓ⁻;组相对归一化;按 seed 重生成噪声逐层更新。
4. 全程无反传、无激活存储;打分前向是并行 teacher forcing,远便宜于自回归生成。

### 为什么这修好了 v1

- L 连续 → 两份扰动的损失几乎永远不同,**估计器级死组消失**。全错组只是 Â=0(与 GRPO 同病,数据层面 DAPO 过滤即可),不再杀死整个更新。
- GRZO 论文的理论(per-example 扰动、B 个有效方向、1/B 方差、收敛界)假设的就是连续损失,**原封不动适用**。v1 失败的根源(信号被 0/1 量化)不存在了。
- 两层归一化各司其职,不再混淆:GRPO 层的组归一化作用于 reward(产生 Â,处理题目难度);GRZO 层的组归一化作用于 δ(ZO 估计器内部,处理方向方差)。v1 把两者揉成了一个。
- σ 的角色退回 ZO 平滑半径(小即可),不再承担探索职责 → 不依赖"σ 球内有决策边界"。

### 卖点定位(诚实版)

不是"比 GRPO 快"——ZO 单步方差仍随有效维度走,收敛速度是主要科学风险。主张是:**在纯推理硬件、推理级显存上到达 GRPO 的优化目标**(超大模型、显存受限、推理专用集群场景),且比 ES 有 per-prompt 信用分配、比 raw-reward ZO 免疫稀疏性。

## 5. 扩展与野心(第二章 / 展望)

- **Angle A(探索扩展)**:rollout 阶段用 K 个扰动模型生成,恢复"轨迹级探索"故事;更新仍走 Angle B 的加权似然。对标 PSN-RLVR:他们扰动探索但反传更新、必须 TIS;我们全 forward-only,σ→0 时 mixture 与 π_θ 偏差 O(σ),有可写的理论。
- **Angle C(野心版,留展望)**:ZO 把"生成→评分"当黑盒,可直接优化分布层面泛函——pass@k(E[max of k])、self-consistency 投票准确率——policy gradient 对这些要么别扭要么没有干净梯度。直接回应"RLVR 只提 pass@1 不扩能力边界"的批评。

## 6. 与 TCAD yield 工作的统一叙事

两个问题是同一类对象:外生噪声 + 指示函数 + 期望才光滑。

| TCAD(yield) | RLVR |
|---|---|
| 设计变量 x | 参数 θ |
| 过程噪声 ξ∼ρ(foundry 固定) | 采样随机数 u(token 抽签) |
| pass/fail 指示函数 | 0/1 reward |
| Yield Y(x) | J(θ) |
| "yield 分段常数、梯度 a.e. 为零" | dead groups |
| softplus margin surrogate ℓ(x,ξ) | Â-加权 log-likelihood 代理 L(θ) |
| per-sample 独立方向 v⁽ⁱ⁾ + CRN | GRZO per-example 扰动 + two_point 共享种子 |

关键差异 = LLM 论文的新颖点:电路的 ρ(ξ) 与 x 无关,**没有 likelihood 通道**,只能挖仿真 margin 并靠 Spearman 校准门保证排序一致;RLVR 的分布 π_θ 本身被优化,log π 通道免费且**梯度精确**(策略梯度恒等式),无需校准。可反向搬运的资产:(i) Spearman 校准门 → "calibrated reward shaping"(有原则的 anti-hacking 检查,Countdown 的 |值−target| 即 margin);(ii) CRN 方差分析 → 支撑 T>0 采样下的两点估计。研究主线统一为:**stochastic indicator objective 的 ZO 优化**(EDA + LLM 两个战场,同一估计器家族)。

## 7. 相关工作定位

| 工作 | 探索 | 更新 | 信号 | 与我们的差异 |
|---|---|---|---|---|
| GRPO | token 采样 | 反传 | Â·∇log π | 我们去掉反传 |
| ES-at-Scale | 参数扰动 | ZO | raw reward,m=200 聚合 | 无 per-prompt 信用分配;m 大、样本贵 |
| PSN-RLVR | 参数扰动 | 反传 + TIS | Â·∇log π | 我们无反传、无 TIS |
| MeZO / GRZO(SFT) | — | ZO | 连续 CE loss | 无 RL 环节;我们把它接进 RLVR |
| v1(本项目,已证伪) | 参数扰动 + greedy | ZO | raw 0/1 reward,m=1 | 死于量化;成为论文 motivation |

## 8. 下一步

1. **查新(最优先,动手前)**:"ZO / forward-only 优化 RLHF·RLVR 代理损失"是否已有人做。目前扫描未见,需系统确认。
2. **实验矩阵(1.5B, Countdown, binary, 同生成预算)**:
   - GRZO-on-surrogate(Angle B,主角);
   - GRPO 反传(上界参照,verl 或 GRPO-Zero);
   - ES-at-Scale 原版(forward-only 参照 + pipeline 正确性对照);
   - v1 binary+G32 跑完留作 motivation 证据。
   - 主指标:test answer_acc vs 生成次数;副指标:峰值显存、δ 方差、dead-prompt 率。
3. **实现要点**:teacher-forcing 打分在 vLLM 里走 prompt_logprobs;扰动打分复用现有 seed 重生成/in-place 机制;注意 ℓ 的数值范围(长序列 NLL 大,考虑按 token 数归一)。
4. **风险清单**:ZO 方差 vs 有效维度(收敛太慢则上 GRZO 的方向数杠杆 / 低秩组合);Â 的噪声;打分前向的额外开销核算;memory 账要算清(rollout 两边都用 vLLM,省的是反传侧的激活+梯度+优化器状态)。

## 9. 遗留操作项(给 coding agent)

- 停掉 0.5B 与 1.5B shaped baseline(曲线已够做图,存好 ckpt/日志)。
- binary+G32 跑完(motivation 数据)。
- 排 ES 原版 positive control。
- σ 探索性诊断降级为可选(Angle B 下 σ 只是平滑半径;Angle A 阶段再捡起来)。
