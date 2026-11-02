# mblisp 之外：moongrad 项目申报书

## 1. 项目名称

**moongrad** —— 基于 MoonBit 的反向模式自动微分引擎

## 2. 项目简介

moongrad 是一个用纯 MoonBit 实现的自动微分引擎。它把一个程序编译成计算图，
再把该程序的**导数**编译成第二张图。梯度因此是一等公民：可以打印、可以统计
节点数、可以在新的输入点重新求值、也可以再求一次导得到 Hessian 矩阵。

与常见的 tape 式实现不同，moongrad 的反向传播**生成程序而不是累加数字**。
每条局部导数规则都以表达式的形式发射到图里（`d/dx sin(x)` 变成一个作用在
原始 `x` 节点上的 `cos` 节点，而不是数值 `cos(1.7)`）。这带来两个直接结果：

- **梯度可重新求值。** tape 式引擎在权重更新后再次索取梯度，返回的是旧权重
  对应的答案；moongrad 返回的是程序，因此在任意绑定下都能重新求值。
- **二阶导数几乎免费。** Hessian 的第 (i, j) 元就是"再对梯度求一次导"，
  而梯度编译器是同一个。`hessian.mbt` 因此只有十一行。

整个项目零外部依赖，只用 MoonBit 标准库，151 个单元测试，编译器 0 warning，
可直接编译到 wasm / wasm-gc / js / native 四个目标。

## 3. 项目方向与通用性

方向：**基础软件层的数学计算内核**（编译器 + 数值计算）。

MoonBit 生态目前缺少一个可嵌入的、wasm 原生的微分计算内核。已有的科学计算
库大多停留在线性代数层面，梯度必须由调用方手写；moongrad 补的是"从标量程序
到梯度程序"这一层，因此它可以直接被上层模型库、参数化建模工具、优化器使用。

通用性不绑定任何具体场景：

- 作为**嵌入式 DSL 引擎**，让业务方用数学表达式而不是代码来表达规则；
- 作为**教学工具**，逐行展示反向传播是怎么工作的；
- 作为**二次型求解器**的底座，Hessian / Newton 步已实现；
- 作为**内存受限环境**下的梯度计算方案，checkpointing 已实现。

## 4. 预期使用场景

### 场景一：参数化建模与曲线拟合

给定带噪声的观测数据，需要反推物理模型的参数。用 moongrad 拟合
`y = 1.5x² - 0.8x + 0.25`，四个特征 `[1, x, x²]` 走一个线性层，训练完直接
读出权重就是多项式系数。`examples/polyfit` 会打印真实系数与拟合系数，并给出
训练区间外的留出点预测（真值 7.625，拟合 7.6299）。

### 场景二：带约束的规则引擎

风控/运维场景里规则常常带耦合，例如"当 A 指标超过阈值且 B 指标连续三次超限
时升级告警级别"。这类规则用 if 嵌套写出来会迅速失控；用 moongrad 可以在
规则里直接写可微表达式，梯度用于自动搜索阈值。

### 场景三：内存受限下的梯度计算

边缘设备或 wasm 沙箱里，完整 tape 的内存占用不可接受。moongrad 的
segment-recomputation checkpointing 把峰值占用从 O(n) 降到 O(k+w)，代价是
多算一遍。`examples/checkpoint` 打印完整的取舍表，并验证每种配置下得到的
梯度逐位相同。

## 5. 拟实现的核心功能

| 模块 | 内容 |
|---|---|
| `node.mbt` | IR：扁平 slot 数组，边始终指向更小的下标 |
| `graph.mbt` | 构建器：hash-consing（公共子表达式共享）、peephole 化简、统计 |
| `simplify.mbt` | 化简规则与局部导数表 |
| `forward.mbt` | 常量传播；另含独立的 forward-mode 对偶数引擎（测试对照用） |
| `reverse.mbt` | 梯度编译器：反向扫描发射导数表达式 |
| `hessian.mbt` | 二阶导数、Cholesky 分解、带阻尼的 Newton 步、Jacobi 特征值 |
| `checkpoint.mbt` | 分段重算的梯度检查点 |
| `tensor.mbt` | 行主序矩阵 |
| `nn.mbt` | 全连接层、激活函数、损失函数、确定性 LCG |
| `optim.mbt` | SGD、动量、Adam（含偏差修正与 eps） |
| `train.mbt` | 训练循环：图与梯度只编译一次，之后每步重新求值 |

语言子集：9 个 special form（`quote` / `if` / `define` / `set!` / `lambda` /
`let` / `let*` / `cond` / `begin`）+ `letrec` + named-let；参数是一等公民，
梯度编译器同时对变量和参数求导，优化器一次拿到两种梯度。

## 6. 项目性质

**原创项目**。解释器与微分引擎均从零实现，未移植或改写任何已有开源项目。
参考资料仅为公开规范与教科书内容：

- R5RS Scheme 报告（语义与标准形式定义）；
- *Automatic Differentiation in Machine Learning: A Review*（Bayingbo 等，
  2018，arXiv:1802.06997）——反向模式与前向模式的划分，以及
  checkpointing 的原始表述；
- Chen et al., *Training Deep Nets with Subgradient Descent with Momentum*,
  arXiv:1412.6980——梯度检查点；
- *Numerical Recipes*——线性同余伪随机数发生器的常数。

上述均为思想与公式层面的公开参考，代码为独立实现。

## 7. 许可与合规

- 许可证：Apache-2.0，仓库根目录附完整 `LICENSE`；
- 依赖：仅 MoonBit 标准库 `moonbitlang/core` 的 `math` 与 `hashmap`，
  无任何第三方 crate；
- CI：GitHub Actions 覆盖 `moon fmt --check`、`moon check`、`moon test`、
  六个 example 的冒烟运行，以及 wasm/wasm-gc/js/native 四目标类型检查；
- 提交历史：16 次以上原子提交，每次提交对应一个可独立验证的变更
  （IR → 构建器 → 化简 → 前向 → 反向 → 二阶 → 检查点 → 张量 → 神经网络
  → 优化器 → 训练循环 → 测试 → 文档 → CI）。