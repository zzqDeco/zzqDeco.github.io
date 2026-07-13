---
title: "Einops 论文精读：Einstein-like Pattern Language、张量形状语义与跨框架执行"
description: "从 ICLR 2022 论文与 einops 0.8.2 源码双线精读 rearrange、reduce、repeat、pack、einsum、Pattern Parser、Transformation Recipe 与多后端执行机制"
pubDate: "2026-07-13T11:04:11+08:00"
updatedDate: "2026-07-13T11:04:11+08:00"
tags:
  - "Paper Reading"
  - "Python"
  - "Deep Learning"
  - "Tensor Programming"
  - "Einops"
  - "Code Reading"
draft: false
---

深度学习代码里最危险的部分，往往不是卷积、注意力或损失函数本身，而是这些模块之间看似平凡的 `reshape`、`transpose`、`view`、`permute`、`squeeze`、`repeat` 和 `concatenate`。一段代码可以顺利运行，输出 shape 也完全正确，却已经把高度、宽度、通道或 attention head 的元素混在一起。由于错误不会触发类型检查、越界异常或 shape mismatch，它可能一直潜伏到训练结束，最终被错误地解释成模型设计、数据质量或超参数问题。

Einops 的价值正是从这个不起眼的薄弱点开始。它没有发明新的张量计算，也没有提供新的 GPU kernel，而是把“输入有哪些轴、输出有哪些轴、哪些轴被组合、分解、消去或新增”写进一条可执行的 pattern。`rearrange(x, 'b c h w -> b h w c')` 不只是 `permute(0, 2, 3, 1)` 的短写；前者把轴的语义、输入契约和输出契约放在同一个局部表达式里，后者只给出了四个位置编号。

本文精读 Alex Rogozhnikov 的 **Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation**，该文发表于 ICLR 2022 并入选 Oral。文章同时对照 einops 稳定版 `v0.8.2` 的源码，固定 commit 为 [`8e911db71f2e693a0c434b041180388c685ed06f`](https://github.com/arogozhnikov/einops/tree/8e911db71f2e693a0c434b041180388c685ed06f)。这很重要：论文时代的核心是 `rearrange / reduce / repeat`；今天常见的 `einsum`、`pack / unpack`、Array API、`torch.compile` 和 MLX backend 都是论文之后的演进，不能倒写成 ICLR 2022 已经完成的能力。

![Einops paper title and abstract](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-title-abstract.png)

*Source: Rogozhnikov, ICLR 2022, title and abstract excerpt via OpenReview archival copy.*

## 1. 一句话贡献

Einops 的一句话贡献是：

> 用一门很小的、可运行时验证的 pattern language，把多维数组的结构变换从位置编号操作改写成命名轴之间的关系。

这句话包含四层含义。

第一，pattern 描述的是**结构**。`'b c h w -> b (h w) c'` 明确表示保留 batch 与 channel，把高度和宽度按 C-order 合成 token 轴。它不是对某个 shape 的一次性算术。

第二，pattern 是**局部接口**。读者不必向前追踪 `x.shape`，也不必向后猜测 `-1` 代表什么。输入和输出的轴同时出现在调用点。

第三，pattern 是**可验证契约**。einops 会检查 rank、轴是否重复、两侧轴集合是否合法、复合轴能否整除，以及新增轴长度是否已提供。

第四，pattern 会被**编译为后端原语**。它最终仍执行 reshape、reduce、transpose、broadcast/stack 等 NumPy、PyTorch、TensorFlow、JAX 或 MLX 操作。einops 的作用类似一个小型结构编译器和跨框架适配层，而不是计算 kernel 编译器。

## 2. 论文信息、版本与许可边界

论文作者只有 Alex Rogozhnikov 一人。OpenReview 页面记录其作为 ICLR 2022 conference paper 发布，ICLR 虚拟会场将其列为 Oral。论文最终 PDF 共 21 页，其中主文到第 9 页，附录包含可读性小实验、多头注意力完整案例、Glow 可逆变换、性能测试、缓存实验和代码规模统计。

本文采用三条相互独立的版本线：

| 线索 | 本文采用的口径 | 不能混写的内容 |
| --- | --- | --- |
| 论文 | ICLR 2022 最终版 | 不把后续 API 倒写进论文贡献 |
| 源码 | `v0.8.2`，commit `8e911db...` | 不跟随 `main` 的开发行为 |
| 开发版 | PyPI `0.9.0.dev0`、仓库 `0.9.0dev` | 只作为未来版本信号，不当稳定 API |

`v0.8.2` 发布于 2026 年 1 月 26 日，完整加入 MLX backend，并把最低 Python 版本提高到 3.9。代码仓库使用 MIT License；论文截图来自 OpenReview 论文，不据此臆造 CC 授权。本文对论文截图只做等比例裁切，并在每张图下保留来源。

复现状态也需要写清：本文使用官方 `v0.8.2` 源码做静态阅读和小规模 NumPy/PyTorch 验证，没有为博客仓库添加 Python 依赖，也没有把 einops 的 benchmark 重新解释成当前硬件上的性能结论。

## 3. “Tensor”一词的三个边界

理解论文前，先把三个经常混在一起的概念分开。

### 3.1 深度学习框架中的 tensor

在 PyTorch、TensorFlow 或 JAX 的日常语境中，tensor 通常指带有 shape、dtype、device 和自动微分语义的 n-dimensional array。一个 shape 为 `(32, 3, 224, 224)` 的对象，可以约定为 `batch, channel, height, width`，但这四个业务含义通常不存储在 tensor 本身。

### 3.2 数学和物理学中的张量

数学张量不是“任意高维数组”的同义词。它具有坐标变换下的特定规律。论文标题有意使用工程社区的 tensor 语义，脚注也向数学和物理背景读者说明可能的术语混淆。Einops 不试图提供微分几何意义上的张量代数。

### 3.3 Einstein summation

Einstein summation convention 用重复指标表达求和，例如矩阵乘法：

$$
C_{ij}=\sum_k A_{ik}B_{kj}.
$$

Einops 借用了“用轴名描述计算关系”的思想，但 `rearrange / reduce / repeat` 并不等同于 `einsum`。论文时期的 einops pattern 主要描述单个张量的结构对应关系；后续加入的 `einops.einsum` 才直接覆盖 tensor contraction，而且仍有自己的语法边界。

因此，最准确的说法是：einops 提供 **Einstein-inspired notation for operations**，不是完整 Einstein calculus。

## 4. 传统张量 API 的问题

论文把主流张量操作概括为三点：tensor 是带 shape 和 dtype 的 n 维数组；不同 tensor 的轴通常按位置对齐；操作通过轴编号或固定布局约定来指定特殊轴。这套接口简单、高效、普及，却把轴语义留在程序员脑中。

![Mainstream tensor operations in the paper](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-mainstream-tensor-operations.png)

*Source: Rogozhnikov, ICLR 2022, mainstream tensor operations excerpt via OpenReview archival copy.*

假设输入 `x` 的 shape 是 `(b, c, h, w)`，希望得到 `(b, h, w, c)`。原生写法可能是：

```python
y = x.permute(0, 2, 3, 1)
```

四个数字都依赖调用点之外的信息。若输入后来改成 `(b, t, c, h, w)`，这行代码需要整体重算。更糟的是，在 reshape 链中，即使写错顺序，元素总数仍相等，程序也能正常运行：

```python
# 两个结果 shape 相同，但只有一个保持了预期图像结构
y1 = x.transpose(2, 0, 3, 1).reshape(h, b * w, c)
y2 = x.transpose(2, 3, 0, 1).reshape(h, b * w, c)
```

这类错误有三个典型特征：

1. 输出 shape 合法。
2. dtype 和 device 不变。
3. 数值看起来仍像“正常的浮点数”。

因此常规单元测试若只断言 shape，也无法识别轴混排。

## 5. Shape Comment 为什么不可靠

工程代码常用注释补救：

```python
# x: [batch, heads, seq, dim]
x = x.permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)
```

注释有价值，但不能替代结构契约。

- 注释不会运行，输入 rank 错误时不会主动报错。
- 注释不会验证 `dim_model % heads == 0`。
- 注释无法保证 `view` 前的 contiguous 条件。
- 注释会随着实现修改而漂移。
- 注释通常只写起点或终点，不描述中间每次 reshape 后的轴关系。

Einops 把同一信息变成调用的一部分：

```python
x = rearrange(x, 'batch heads seq dim -> batch seq (heads dim)')
```

它仍不能证明 `heads` 在业务上真的是 attention head，但至少能证明输入输出轴之间满足该结构关系。

## 6. 论文 Figure 1：shape 正确仍可破坏结构

论文 Figure 1 用图像张量展示了最具说服力的失败案例。同一个底层元素序列，在两个合法 reshape 中可以得到完全不同的结构：一个只混合局部颜色，另一个把所有轴打散。结果 shape 都合法，程序不会报错。

![Breaking tensor structure with reshape](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-figure1-broken-tensor-structure.png)

*Source: Rogozhnikov, ICLR 2022, Fig. 1 via OpenReview archival copy.*

这张图支持的是“位置 API 容易让结构错误静默通过”，不是“所有 reshape 都危险”。reshape 本身是必要且高效的原语。问题是裸 reshape 没有表达原轴如何对应到新轴：

$$
\prod_{i=1}^{m} n_i = \prod_{j=1}^{k} n'_j
$$

只保证元素总数相等，并不保证 `height`、`width`、`channel` 的语义关系正确。

## 7. 论文的设计目标

论文提出的 notation 可归纳为五个设计目标。

### 7.1 Semantic information

轴名应表达结构角色，如 `batch`、`height`、`width`、`channel`、`head`、`token`。它们靠近操作，而不是只存在于注释和变量命名里。

### 7.2 Runtime checks

操作应检查 rank、轴集合、已知长度和整除关系。错误应尽可能在结构变化处暴露。

### 7.3 Strictly determined result

给定输入 shape、pattern 与显式轴长度，输出 shape 和元素排列应被唯一决定。pattern 不是提示，而是执行定义。

### 7.4 Uniformity

转置、展平、拆轴、合轴、stack、concatenate、pooling、repeat 应共享同一种轴语言，而不是每个操作各有一套位置参数。

### 7.5 Framework-independent behavior

同一 pattern 应在不同后端表达相同的结构变换，后端差异由 adapter 处理。

## 8. Pattern 基本语法

一个典型 pattern 由左侧输入表达式、箭头和右侧输出表达式组成：

```text
input axes -> output axes
```

例如：

```python
from einops import rearrange

# input:  (batch=8, channel=3, height=32, width=32)
# output: (batch=8, height=32, width=32, channel=3)
y = rearrange(x, 'batch channel height width -> batch height width channel')
```

单个标识符代表一个 elementary axis；空格分隔轴；括号把多个 elementary axes 合成一个 composite axis。轴名不存入 tensor，它只在本次操作中建立对应关系。

![Einops pattern language](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-pattern-language.png)

*Source: Rogozhnikov, ICLR 2022, pattern-language excerpt via OpenReview archival copy.*

轴名是语义工具，不是静态类型。把 channel 错写成 height，只要长度和集合仍合法，einops 无法知道业务含义写错了。这也是后文必须讨论的边界。

## 9. 轴组合、分解与 C-order

组合轴是 einops notation 最关键的能力：

```python
# input:  (b, c, h, w)
# output: (b, h*w, c)
tokens = rearrange(images, 'b c h w -> b (h w) c')
```

括号表示右侧的 token 轴由 `h` 和 `w` 组成。采用 C-order 时，最右边的轴变化最快。若把 `(i1, i2, i3)` 合成一个索引，线性位置为：

$$
i=((i_1 n_2)+i_2)n_3+i_3.
$$

所以 `(h w)` 与 `(w h)` 元素顺序不同，即使输出长度都是 $hw$。这是 review pattern 时必须关注的地方。

分解轴则需要足够的长度信息：

```python
# x.shape == (8, 196, 768)
# output == (8, 14, 14, 768)
grid = rearrange(x, 'b (h w) c -> b h w c', h=14)
```

`h=14` 后，`w` 可由 `196 / 14` 推断。若 `196` 不能整除 `14`，einops 立即抛出 `EinopsError`。若 `h` 和 `w` 都未知，则分解不是唯一的：196 可以是 `14×14`、`7×28`、`4×49`，库不能凭空选择。

## 10. Anonymous Axis、Singleton 与空组合

数字可以表示 anonymous axis：

```python
# 每个 width 位置复制 3 次
y = repeat(x, 'h w -> h (w 3)')
```

但数字 `1` 具有特殊 singleton 语义：

```python
# input: (h, w) -> output: (1, h, w, 1)
y = rearrange(x, 'h w -> 1 h w 1')
```

论文语法中空组合 `()` 也表示单位轴：

```python
# input: (1, h, w) -> output: (h, w)
y = rearrange(x, '() h w -> h w')
```

这些写法不能不分操作地滥用。`rearrange` 必须保持元素一一对应；非单位 anonymous axis 若凭空出现在输出侧，就改变元素数量，应由 `repeat` 表达。相反，输入侧独有的非单位轴意味着 reduction，应由 `reduce` 表达。把三类操作分开，是为了让错误信息和语义边界更清晰。

## 11. Ellipsis 与 rank polymorphism

`...` 表示零个或多个未命名轴：

```python
# 交换最后两个轴，前面任意数量的轴保持不变
y = rearrange(x, '... h w -> ... w h')
```

这允许同一 pattern 处理 `(h, w)`、`(b, h, w)`、`(b, t, h, w)`。它是一种运行时 rank polymorphism，但不是完整类型多态。

几个约束很重要：

1. 同一表达式中 ellipsis 不能重复到产生歧义。
2. `rearrange` 右侧出现 ellipsis 时，左侧也必须定义它。
3. 括号中的 ellipsis 代表把若干轴组合起来，需要满足 parser 对该 operation 的限制。
4. 可变 rank 会提高复用性，也会降低局部可见性；固定 rank 的核心模型代码不一定要用 ellipsis。

## 12. 三种核心操作的语法不变量

论文选择三个独立函数，而不是一个万能 `transform()`，是为了把非法结构变换尽早排除。

| 操作 | 左右两侧轴关系 | 语义 |
| --- | --- | --- |
| `rearrange` | elementary axes 集合相同 | 保持元素，一一重排 |
| `reduce` | 右侧是左侧子集 | 消去轴并聚合 |
| `repeat` | 左侧是右侧子集 | 新增轴并复制/broadcast |

用集合直觉表示：

$$
A_L=A_R \quad (rearrange),
$$

$$
A_R\subseteq A_L \quad (reduce),
$$

$$
A_L\subseteq A_R \quad (repeat).
$$

实际实现还要处理 composite axis、数字轴、ellipsis 和单位轴，所以不是简单 set comparison，但这三个不变量是理解错误信息的最佳入口。

## 13. `rearrange`：结构等价变换

`rearrange` 统一表达 transpose、reshape、flatten、squeeze、unsqueeze、stack 和 concatenate 的常见组合。

### 13.1 Transpose

```python
# (b, c, h, w) -> (b, h, w, c)
y = rearrange(x, 'b c h w -> b h w c')
```

### 13.2 Flatten spatial axes

```python
# (b, c, h, w) -> (b, h*w, c)
y = rearrange(x, 'b c h w -> b (h w) c')
```

### 13.3 Stack a list

```python
# 4 tensors of shape (b, c, h, w) -> (b, 4, c, h, w)
y = rearrange([x1, x2, x3, x4], 'views b c h w -> b views c h w')
```

### 13.4 Split a dimension

```python
# (b, tokens, heads*dim) -> (b, heads, tokens, dim)
y = rearrange(x, 'b tokens (heads dim) -> b heads tokens dim', heads=8)
```

`rearrange` 在可能时返回 view，但“可能”取决于后端和内存布局。组合连续轴通常只需 reshape；改变非连续轴顺序可能需要 transpose view；后续要求 contiguous 的算子可能触发 copy。不能把 einops 语法简洁推导成零拷贝保证。

## 14. `reduce`：命名轴聚合

`reduce` 把 pooling 和统计聚合写成轴消失：

```python
from einops import reduce

# global average pooling: (b, c, h, w) -> (b, c)
pooled = reduce(x, 'b c h w -> b c', 'mean')

# 2x2 max pooling: (b, c, h*2, w*2) -> (b, c, h, w)
pooled = reduce(
    x,
    'b c (h kh) (w kw) -> b c h w',
    'max',
    kh=2,
    kw=2,
)
```

设被消去轴集合为 $R$，一般 reduction 可写作：

$$
y_{A_R}=\rho_{A_L\setminus A_R}x_{A_L},
$$

其中 $\rho$ 可以是 `sum`、`mean`、`max`、`min`、`prod`、`all`、`any`，也可以是 callable。callable reduction 提供扩展点，但跨 backend 行为取决于 callable 是否理解对应 tensor 类型。

一个常见错误是把“先 reshape 再 max”写成只对某个位置轴 max。pattern 会把 kernel 轴 `kh, kw` 和输出空间轴 `h, w` 分开，使 pooling 语义更直观。

## 15. `repeat`：新增轴与复制语义

NumPy 的 `repeat`、`tile`，PyTorch 的 `repeat`、`repeat_interleave` 和 broadcast 语义容易混淆。einops 通过输出轴位置直接表达目标：

```python
from einops import repeat

# (h, w) -> (b, h, w, c)
y = repeat(x, 'h w -> b h w c', b=8, c=3)

# 每个位置连续重复 r 次
y = repeat(x, 'h w -> h (w r)', r=2)

# 整个 width 序列复制 r 组
y = repeat(x, 'h w -> h (r w)', r=2)
```

后两条输出 shape 相同，但顺序不同。`(w r)` 中 `r` 是快速变化的内层轴，表现为每个元素连续重复；`(r w)` 中整段 `w` 被重复。这正是 named composite axes 比 `repeat(2)` 更有解释力的地方。

## 16. `pack / unpack`：后论文时代的可逆拼接

`pack / unpack` 在 0.6 版本进入 einops，不属于 ICLR 2022 论文的核心 API。它们解决的不是“已知每个轴长度如何重排”，而是“若干 tensor 在保留公共轴后，如何把剩余任意 rank 的部分打包，并保存可逆元数据”。

```python
from einops import pack, unpack

# class_token: (b, c)
# image_tokens: (b, h, w, c)
# text_tokens: (b, t, c)
sequence, packed_shapes = pack(
    [class_token, image_tokens, text_tokens],
    'b * c',
)
# sequence: (b, 1 + h*w + t, c)

class_out, image_out, text_out = unpack(
    sequence,
    packed_shapes,
    'b * c',
)
```

`*` 是唯一 wildcard。`pack` 将 wildcard 覆盖的轴展平后拼接，同时返回每个输入的原始 packed shape；`unpack` 用这些 shape 计算切片并恢复 rank。若 packed 长度为 $p_i$，总长度为：

$$
P=\sum_i p_i,
\qquad
p_i=\prod_j n_{ij}.
$$

这比手写 `flatten + cat + split + reshape` 更可靠，因为恢复所需元数据由前向操作产生，而不是在另一处重新计算。

## 17. `einsum`：多字符轴名的 contraction

`einops.einsum` 在 0.5 版本加入，也晚于论文最初核心设计。它与 NumPy/PyTorch einsum 的主要区别是轴名可以是完整单词，并且 pattern 放在所有 tensor 参数之后：

```python
from einops import einsum

# q: (batch, query, head, dim)
# k: (batch, key, head, dim)
# scores: (batch, head, query, key)
scores = einsum(
    q,
    k,
    'batch query head dim, batch key head dim -> batch head query key',
)
```

对应公式：

$$
s_{bhqk}=\sum_d q_{bqhd}k_{bkhd}.
$$

它不完全复用 `rearrange` grammar。当前 `einsum` 不支持 composite axes 和 singleton `()` 语法；重复轴代表 contraction 或 diagonal，语义也与 `rearrange` 的“一侧轴唯一”不同。统一轴命名不代表所有 operation 共享完全相同的 parser 规则。

## 18. `parse_shape` 与 `asnumpy`

`parse_shape` 用 pattern 把 shape 映射为字典：

```python
from einops import parse_shape

shape = parse_shape(x, 'batch channel height width')
# {'batch': 8, 'channel': 3, 'height': 224, 'width': 224}
```

它适合把外部 API 仍要求整数参数的地方接到 named-axis 代码上，但不要用它重新制造大段 shape 算术。若一段代码先 `parse_shape`，再用字典拼出多个 reshape，通常说明结构仍可直接写成 pattern。

`asnumpy` 则把受支持 backend 的 tensor 转为 NumPy array。它是小型互操作工具，不负责 device placement、梯度保留或零拷贝承诺。对 PyTorch CUDA tensor，转换必然涉及 detach/CPU 等 backend 处理，不能把它当训练图中的透明操作。

## 19. Framework Layers：把 pattern 放进模型结构

函数式 API 适合 forward 中即时变换；`einops.layers` 则允许把相同语义放入 `Sequential` 或可序列化模型配置：

```python
from torch import nn
from einops.layers.torch import Rearrange, Reduce

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    Rearrange('b c h w -> b (h w) c'),
    nn.Linear(64, 128),
    Reduce('b tokens c -> b c', 'mean'),
)
```

Layer API 与函数式 API 的 shape 语义相同，但生命周期不同。layer 在构造时保存 pattern 和轴长度，框架可以把它纳入 `state_dict`、模型打印、JIT 或配置序列化。它通常没有可训练参数，却是网络拓扑的一部分。

不要为了“统一风格”把 forward 中每个临时变换都改成 layer。选择标准应是：该变换是否属于模型结构、是否需要被序列化或出现在 module tree 中。

## 20. EinMix：把线性映射的轴语义写出来

`EinMix` 是 einops layers 中更接近可训练计算的组件。它用 pattern 描述输入轴到输出轴的变换，再用 `weight_shape` 和可选 `bias_shape` 指明参数依赖哪些轴。

一个按通道执行的线性层可写为：

```python
from einops.layers.torch import EinMix

mix_channels = EinMix(
    'batch height width channel_in -> batch height width channel_out',
    weight_shape='channel_in channel_out',
    bias_shape='channel_out',
    channel_in=64,
    channel_out=128,
)
```

其核心公式是：

$$
Y_{b h w o}
=\sum_i X_{b h w i}W_{io}+b_o.
$$

MLP-Mixer 的 token mixing 和 channel mixing 也可分别写出参数在哪个轴共享：

```python
token_mixing = EinMix(
    'b tokens channels -> b mixed_tokens channels',
    weight_shape='tokens mixed_tokens',
    bias_shape='mixed_tokens',
    tokens=196,
    mixed_tokens=196,
)
```

`EinMix` 的优势不是替代所有 `Linear`，而是在高 rank 输入上显式表达“哪些轴参与映射、哪些轴只是 batch-like 地独立处理”。源码入口在 [`einops/layers/_einmix.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/layers/_einmix.py)；不同框架 layer 再把抽象参数落到各自 module 类型。

## 21. Case Study：Vision Permutator

论文主文的第一个 case study 是 Vision Permutator，不是 ShuffleNet。原代码要在 height mixing 与 width mixing 之间同时修改多次 reshape、permute 和 projection 前后布局；einops 版本只需要交换 pattern 中的 `h` 和 `w`。

![Vision Permutator case study](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-vision-permutator-case-study.png)

*Source: Rogozhnikov, ICLR 2022, Vision Permutator case-study excerpt via OpenReview archival copy.*

核心差异可以压缩为：

```python
# mix along height
x_h = rearrange(x, 'b h w (n s) -> b n w (h s)', s=s)
x_h = rearrange(proj_h(x_h), 'b n w (h s) -> b h w (n s)', s=s)

# mix along width
x_w = rearrange(x, 'b h w (n s) -> b n h (w s)', s=s)
x_w = rearrange(proj_w(x_w), 'b n h (w s) -> b h w (n s)', s=s)
```

读者能直接看到：第一次把 `h` 合入 projection feature，第二次把 `w` 合入。修改某条路径时，另一条的轴名不会因为整数位置整体偏移而连锁变化。

这不是形式上的“少写两行”。它改变了 review 单位：reviewer 可以逐个比较输入和输出 pattern，而不是在脑中模拟 `reshape -> permute -> reshape` 的位置变换。

## 22. Case Study：ShuffleNet Channel Shuffle

ShuffleNet 的 channel shuffle 是官方示例和论文附录代码规模统计中的典型工程案例，但不是论文主文独立 case study。正确区分来源后，它仍非常适合展示 composite axis。

原生 PyTorch 常写成：

```python
batch, channels, height, width = x.shape
channels_per_group = channels // groups
x = x.view(batch, groups, channels_per_group, height, width)
x = x.transpose(1, 2).contiguous()
x = x.view(batch, channels, height, width)
```

einops 版本：

```python
x = rearrange(
    x,
    'batch (groups channels_per_group) height width '
    '-> batch (channels_per_group groups) height width',
    groups=groups,
)
```

它明确表示：先把 channel 分解成 `groups × channels_per_group`，再交换二者顺序并重新合成。两侧元素数保持不变；若 channels 不能被 groups 整除，错误在此处暴露。

## 23. Case Study：Glow Squeeze 与逆变换

Glow 的 squeeze 把空间分辨率换成通道数，是 normalizing flow 中常见的可逆结构变换。原始实现包含多次 reshape 和 transpose，逆变换还要重新推导另一套位置编号。

![Glow reversible squeeze and unsqueeze](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-glow-reversible-rearrangement.png)

*Source: Rogozhnikov, ICLR 2022, Glow squeeze/unsqueeze excerpt via OpenReview archival copy.*

einops 可以把正向与逆向写成交换箭头两侧：

```python
def squeeze2d(x, factor=2):
    return rearrange(
        x,
        'b (h fh) (w fw) c -> b h w (c fh fw)',
        fh=factor,
        fw=factor,
    )

def unsqueeze2d(x, factor=2):
    return rearrange(
        x,
        'b h w (c fh fw) -> b (h fh) (w fw) c',
        fh=factor,
        fw=factor,
    )
```

交换两侧能表达逆变换，前提是所有 elementary axes 都保留、分解长度可推断、没有 reduction/repeat、C-order 一致。最稳妥的测试不是只断言 shape，而是：

```python
assert torch.equal(x, unsqueeze2d(squeeze2d(x)))
```

并使用非对称 shape 与可识别数值，避免两个同长度轴交换后测试仍误通过。

## 24. Case Study：Vision Transformer Patchify

ViT patchify 是 einops 最常见的现代案例之一。给定 `(b, c, h*ph, w*pw)` 图像，把每个 `ph × pw` patch 展平成 token：

```python
patches = rearrange(
    images,
    'b c (h ph) (w pw) -> b (h w) (ph pw c)',
    ph=16,
    pw=16,
)
```

输入 `(2, 3, 224, 224)` 时，输出为 `(2, 196, 768)`。过程可以分成两个独立的结构决定：

1. patch grid 的 `h, w` 合成 token 轴。
2. patch 内的 `ph, pw, c` 合成 feature 轴。

逆变换：

```python
restored = rearrange(
    patches,
    'b (h w) (ph pw c) -> b c (h ph) (w pw)',
    h=14,
    w=14,
    ph=16,
    pw=16,
    c=3,
)
```

Round-trip test 会验证元素顺序，不只是 shape。这里尤其要注意 `(ph pw c)` 与 `(c ph pw)` 不等价；前者匹配目标 token feature 的序列化顺序，后者会产生另一种合法排列。

## 25. Case Study：Multi-Head Attention

论文附录把一个 58 行左右、依赖大量 `view/permute/contiguous` 和 shape comment 的多头注意力实现，与一版轴名清晰的实现并列。论文版本使用 `torch.einsum`，今天也可以用 `einops.einsum` 的多字符轴名进一步提高可读性。

![Multi-head attention case study](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-multi-head-attention-case-study.png)

*Source: Rogozhnikov, ICLR 2022, multi-head-attention case-study excerpt via OpenReview archival copy.*

现代化简化版本如下：

```python
from einops import einsum, rearrange

# q, k, v: (batch, tokens, heads * dim)
q = rearrange(q, 'b query (head dim) -> b head query dim', head=heads)
k = rearrange(k, 'b key   (head dim) -> b head key   dim', head=heads)
v = rearrange(v, 'b key   (head dim) -> b head key   dim', head=heads)

scores = einsum(
    q,
    k,
    'b head query dim, b head key dim -> b head query key',
) / (q.shape[-1] ** 0.5)

attention = scores.softmax(dim=-1)
context = einsum(
    attention,
    v,
    'b head query key, b head key dim -> b head query dim',
)
output = rearrange(context, 'b head query dim -> b query (head dim)')
```

轴名揭示了 contraction：`dim` 在 QK 相乘中消失，`key` 在 attention 与 V 相乘中消失。`query` 与 `head` 保留。实现仍需要 mask、dropout、数值稳定和 grouped-query 等工程处理，但 shape choreography 已更容易审查。

## 26. Case Study：多模态 Transformer 的 token 打包

多模态模型经常需要拼接不同 rank 的 token：class token 是 `(b, c)`，图像 feature 是 `(b, h, w, c)`，文本是 `(b, t, c)`。手写代码要保存每段长度、flatten 图像、concat，再按长度 split 和 reshape。

`pack / unpack` 把这一过程变成一对操作：

```python
sequence, ps = pack(
    [cls_token, image_grid, text_tokens, task_token],
    'batch * channel',
)

sequence = transformer(sequence)

cls_out, image_out, text_out, task_out = unpack(
    sequence,
    ps,
    'batch * channel',
)
```

关键不是少写 split 长度，而是 `ps` 与本次输入绑定。图像分辨率或文本长度动态变化时，恢复信息不会与另一个模块中的硬编码常量漂移。

## 27. 源码总览：从字符串到后端原语

稳定版 `v0.8.2` 的核心执行链可概括为：

```text
public API
  -> identify backend and input rank
  -> parse left/right expressions
  -> validate operation-specific invariants
  -> compile TransformRecipe
  -> reconstruct concrete axis lengths from runtime shape
  -> apply reshape / transpose / reduce / add_axes / reshape
  -> return backend-native tensor
```

源码入口集中在 [`einops/einops.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py)。`rearrange` 实际以特殊 reduction 类型调用通用 `reduce` 路径；`repeat` 同样复用 recipe compiler。这样 parser、shape inference 与执行逻辑不会为三个 API 各复制一遍。

## 28. Pattern Parser：`ParsedExpression`

[`einops/parsing.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/parsing.py#L10-L133) 里的 `ParsedExpression` 解析箭头某一侧，例如 `'b c (h w)'`。它产生的关键结构包括：

- `identifiers`：出现过的 elementary axes 集合。
- `composition`：每个 tensor dimension 由哪些 elementary axes 组成。
- `has_ellipsis`：是否出现 `...`。
- `has_ellipsis_parenthesized`：ellipsis 是否位于 composite axis 内。
- `has_non_unitary_anonymous_axes`：是否有大于 1 的数字轴。

parser 是刻意简单的字符扫描器，不是通用表达式解释器。它只接受字母数字、下划线、括号、空格和 ellipsis；不允许嵌套括号；括号不平衡、重复标识符、非法 Python identifier 会立即报错。

`AnonymousAxis` 的每个实例彼此不等，即使数字相同。这体现了一个细节：两个都写为 `2` 的 anonymous axes 不应自动被认为是同一个语义轴。它们只有长度相同，没有名字对应关系。

## 29. `_prepare_transformation_recipe()`：编译结构计划

[`_prepare_transformation_recipe()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py#L290-L459) 是 pattern compiler 的主体。输入是 pattern、operation、补充轴名和输入 rank，输出是 `TransformRecipe`。

它先解析左右表达式，再执行 operation-specific validation：

```python
# 概念化伪代码，不是逐行复制源码
if operation == 'rearrange':
    require(left_axes == right_axes)
elif operation == 'repeat':
    require(left_axes <= right_axes)
    require(length_is_given_for_new_axes)
elif operation in reductions:
    require(right_axes <= left_axes)
```

随后它把 ellipsis 展开为与输入 rank 对应的内部轴，记录 elementary axis 的已知/未知长度，计算第一次 reshape 后的轴顺序、需要 reduction 的尾部轴、需要新增的轴，以及最终 composite axes。

这一步只依赖 pattern、operation、补充轴名和 rank，不依赖具体 batch size。因此可以被第一级缓存复用。

## 30. `TransformRecipe`：结构计划的数据表示

[`TransformRecipe`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py#L114-L152) 不是可执行 graph，而是一组足以重建执行参数的静态字段：

| 字段 | 含义 |
| --- | --- |
| `elementary_axes_lengths` | 已知长度或待推断占位 |
| `axis_name2elementary_axis` | 轴名到内部 axis id 的映射 |
| `input_composition_known_unknown` | 每个输入维度中的已知/未知 elementary axes |
| `axes_permutation` | elementary axes 的重排顺序 |
| `first_reduced_axis` | reduction axes 起点 |
| `added_axes` | 新增轴的位置与内部 id |
| `output_composite_axes` | 最终每个输出维度由哪些轴组成 |

这种设计把“语法与结构关系”同“这一次输入 shape 的具体数值”分开。相同 pattern 处理 batch 8 和 batch 64 时，共用 recipe，只需重建 concrete shape 参数。

## 31. Shape Reconstruction：已知轴、未知轴与整除检查

[`_reconstruct_from_shape_uncached()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py#L155-L225) 接收 recipe、运行时 shape 和调用方提供的轴长度。

对一个输入 composite dimension，它计算已知轴长度乘积：

$$
P_{known}=\prod_{i\in K}n_i.
$$

若没有未知轴，则要求输入维度 $L=P_{known}$。若正好有一个未知轴，则要求：

$$
L\bmod P_{known}=0,
\qquad
n_{unknown}=L/P_{known}.
$$

recipe 构建阶段已经保证每个 composite input dimension 最多有一个真正未知轴。否则无法唯一推断。随后函数构造：初始 reshape shape、transpose permutation、reduced axes、added axes、final reshape shape。

对 TensorFlow、JAX 或 PyTorch symbolic shape，长度可能不是普通 Python int。实现尽量保留符号算术；只有在值确实是 int 时才做直接相等或取模检查。这解释了为什么源码不能只按 NumPy 整数 shape 来理解。

## 32. 执行流水线：最多几类基础操作

[`_apply_recipe()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py#L230-L252) 的主流程非常短：

```text
1. initial reshape：把 composite input dimensions 拆成 elementary axes
2. transpose：把 elementary axes 排到目标/归约顺序
3. reduce：对需要消失的轴聚合
4. add_axes：插入并扩展 repeat 新轴
5. final reshape：把目标 elementary axes 合成 composite output dimensions
```

源码实际顺序是 initial reshape、transpose、reduce、add axes、final reshape。论文概述有时用“reshape-transpose-reshape”解释 rearrange 的典型路径；对于三个通用 operation，完整执行链还必须容纳 reduce 和 repeat。

这说明复杂 notation 并没有制造复杂 runtime。大部分开销仍是后端张量操作本身；Python 层主要负责解析、检查和参数重建。

## 33. Transformation Optimization：消除不必要步骤

recipe 并不会机械执行五步。若输入维度已经是 elementary axis，`init_shapes` 为 `None`；若 permutation 是 identity，跳过 transpose；没有 reduction 或 added axis 时对应步骤为空；输出无 composite axis 时跳过 final reshape。

此外，源码会尝试合并连续、顺序一致的轴，减少中间 shape 的复杂度。这里的“optimization”是对 transformation plan 的代数化简，不是算子 fusion、CUDA codegen 或 kernel autotuning。

例如：

```python
y = rearrange(x, 'a b c d -> (a b) (c d)')
```

若底层布局连续且轴顺序不变，通常可直接 reshape 成两个维度。若 pattern 改成 `'a b c d -> (a c) (b d)'`，必须先改变元素顺序，单次 view 无法完成。

## 34. 两级缓存机制

`v0.8.2` 有两个关键 LRU cache。

第一级在 `_prepare_transformation_recipe()` 上，容量 256，key 包含 pattern、operation、轴名和 ndim。它缓存 parser 与结构编译结果。

第二级在 `_reconstruct_from_shape()` 上，容量 1024，key 进一步包含运行时 shape 和轴长度。它缓存 concrete reshape、permutation 和输出 shape。

若 shape 或轴长度不可 hash，例如某些符号对象，源码捕获 `TypeError` 并回退到 uncached reconstruction，而不是破坏符号执行。

论文缓存实验故意构造 10,000 个不同 pattern，让缓存无法命中，再与重复同一 pattern 比较。结果约为 `663 ms` 对 `37.1 ms`。这个十几倍差异证明 parser cache 对极小数组的 Python 开销很重要，但不能解释成真实神经网络整体加速十几倍。真实模型中卷积、矩阵乘法和 attention kernel 通常占主导。

![Einops caching and applicability study](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-caching-and-flexibility.png)

*Source: Rogozhnikov, ICLR 2022, caching and applicability excerpt via OpenReview archival copy.*

## 35. Backend Dispatch：按 tensor 类型惰性识别

[`get_backend()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/_backends.py#L22-L64) 不会在 import einops 时主动 import 所有框架。它维护 tensor type 到 backend instance 的缓存；若没有命中，再遍历 backend subclasses，并只考虑其框架模块已经出现在 `sys.modules` 的实现。

这种策略解决三个问题：

1. 避免仅使用 NumPy 时加载 PyTorch/TensorFlow 的时间和内存。
2. 避免可选框架安装损坏时，import einops 被连带拖垮。
3. 稳定命中后只需按 type 查表，不必每次扫描所有 backend。

论文用多个框架不同的 transpose API 展示适配价值：`np.transpose`、`tf.transpose`、Keras `permute_dimensions`、PyTorch `permute` 参数形式各不相同，einops pattern 保持一致。

![Backend API comparison](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-backend-api-comparison.png)

*Source: Rogozhnikov, ICLR 2022, backend API comparison excerpt via OpenReview archival copy.*

## 36. Backend Adapter：最小接口而不是框架模拟器

`_backends.py` 为 NumPy、JAX、PyTorch、CuPy、TensorFlow/Keras、OneFlow、Paddle、tinygrad、PyTensor 和 MLX 提供 adapter。每个 adapter 实现 shape、reshape、transpose、reduce、stack/concat、add_axes、einsum、to/from NumPy 等所需最小原语。

Adapter 并不试图统一全部框架 API。它只服务 einops 操作集合。这种克制非常关键：库若尝试抽象随机数、卷积、设备、自动微分、分布式等所有能力，很快会成为另一个大型框架，并不断泄漏最低公分母限制。

后端仍然可能泄漏差异：

- view/copy 语义不同。
- symbolic shape 支持不同。
- callable reduction 可接受的参数不同。
- JIT/compile 对 Python 路径的容忍度不同。
- dtype 和布尔 reduction 行为由后端决定。

所以“framework-independent notation”应理解为结构语义一致，不是所有运行时细节完全相同。

## 37. Array API 路径

`einops.array_api` 是后论文版本对 Python Array API standard 的支持。它与传统 adapter dispatch 有本质区别：传统路径先识别具体 tensor 类型，再调用 einops 自己的 `AbstractBackend`；Array API 路径通过 `__array_namespace__` 或显式 namespace 使用标准化的 `xp.reshape`、`xp.permute_dims`、`xp.expand_dims`、`xp.broadcast_to` 等接口。

[`_apply_recipe_array_api()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py#L255-L287) 把 recipe 直接翻译到 `xp` 原语。它减少专有 adapter 代码，但 Array API 本身覆盖面与各框架实现成熟度仍决定可用边界。

不要写成“0.7 以后所有 backend 都只走 Array API”。稳定版仍同时保留传统 backend adapters 和独立 Array API 模块，两条路径服务不同对象与兼容目标。

## 38. Torch 特殊路径：JIT、`torch.compile` 与符号 shape

PyTorch 是 einops 使用最广、编译边界也最复杂的后端之一。动态 `get_backend()` 和 Python parser 不适合直接交给旧式 TorchScript，所以 [`_torch_specific.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/_torch_specific.py) 提供了两类特殊处理。

### 38.1 Scriptable layers

`TorchJitBackend` 是静态、受限的 backend，只实现 layer script 所需的 reshape、permute、reduce、stack、repeat/expand 等操作。`apply_for_scriptable_torch()` 镜像 `_apply_recipe()`，但绕过运行时 backend discovery，并直接调用 uncached shape reconstruction。

这解释了一个常见现象：函数式 `rearrange()` 与 `einops.layers.torch.Rearrange` 在普通 eager 模式下行为一致，但面对旧 TorchScript 时，layer 版本更容易被脚本化，因为 recipe 在 module 构造阶段已经准备好。

### 38.2 `torch.compile`

`allow_ops_in_compiled_graph()` 对 PyTorch 2.0 到 2.7 使用 `torch._dynamo.allow_in_graph` 注册 `rearrange / reduce / repeat / einsum / pack / unpack`。源码同时注明 PyTorch 2.8 及以上不再需要这一步，因此函数直接返回。

这不是“einops 自己完成图编译”。它只是告诉 Dynamo 这些函数可以进入编译图，随后仍由 PyTorch 分析其底层操作、符号 shape 和 fusion 机会。不同 PyTorch 版本的 graph break 行为必须用目标环境验证，不能只凭 README 宣称无条件兼容。

### 38.3 Symbolic shape

如果 batch、sequence 或图像尺寸以 `SymInt` 等符号表示，shape reconstruction 应尽量保留符号算术。缓存可能因对象不可 hash 而回退，但结构计算仍可执行。团队若同时使用 `torch.compile(dynamic=True)` 与复杂 composite axes，应该把多个动态 shape 加入测试矩阵，而不是只验证一个固定输入。

## 39. View、Copy 与 Contiguous：不能承诺零拷贝

官方文档谨慎地说 `rearrange` 在可能时返回 view。决定因素至少有三层。

### 39.1 Pattern 是否保持可 view 的线性顺序

`'b c h w -> b c (h w)'` 对 contiguous NCHW tensor 通常只需改变 shape metadata。`'b c h w -> b (h w) c'` 先把 channel 移到最后，再合并空间轴，通常需要 transpose；结果可能是非 contiguous view，后续 reshape 或 kernel 可能触发 copy。

### 39.2 输入本身的 stride

切片、transpose、channels-last 或 memory-mapped array 会改变 stride。相同 pattern 应用于不同 stride 的输入，copy 行为可能不同。

### 39.3 Backend 的 reshape 语义

NumPy `reshape`、PyTorch `reshape/view`、JAX functional array 和 TensorFlow graph tensor 对 view 的定义并不完全相同。einops adapter 调用后端原语，无法创建跨框架统一的 storage alias contract。

因此性能敏感代码应以 profiler、stride/contiguous 检查和目标编译器 graph 为准。不要因为 pattern 看起来只是“重命名轴”就假定零内存流量。

## 40. 错误检查能捕获什么

Einops 的可靠性来自一组明确但有限的运行时检查。

### 40.1 Rank mismatch

```python
# 输入是 3D，但 pattern 要求 4D
rearrange(x, 'b c h w -> b h w c')
```

会报告期望和实际维度数。

### 40.2 轴集合不一致

```python
# rearrange 不允许 channel 消失
rearrange(x, 'b c h w -> b h w')
```

应改为明确的 reduction，或确认真正需要的操作。

### 40.3 重复轴和非法标识符

`'b h h -> b h'`、不平衡括号、嵌套括号、非法字符和含糊 ellipsis 会在 parser 阶段失败。

### 40.4 Composite axis 无法整除

```python
# tokens=197，无法按 h=14 拆成 h*w
rearrange(x, 'b (h w) c -> b h w c', h=14)
```

会报告不能把长度 197 按 14 分块。

### 40.5 Repeat 新轴未给长度

```python
repeat(x, 'h w -> batch h w')
```

`batch` 不在输入侧，必须提供 `batch=...`。

这些检查比裸 reshape 更强，因为它们建立在 axis relationship 上，而不仅是元素总数。

## 41. 错误检查捕获不了什么

同样重要的是明确 einops 不能证明什么。

### 41.1 轴名的业务含义

```python
# 实际输入是 b, width, height, channel，调用者却这样命名
y = rearrange(x, 'b height width channel -> b channel height width')
```

若 height 和 width 长度不同，输出仍合法，只是语义错了。轴名是程序员提供的声明，不是 tensor 自带标签。

### 41.2 跨操作的一致性

一个函数用 `tokens`，下一个函数误把同一轴当 `heads`，einops 不会跨调用追踪。它不是 whole-program shape type checker。

### 41.3 同长度轴交换

若 `height == width` 或 `query == key`，测试只比较 shape 很容易漏掉交换。应使用非对称 shape 或带坐标编码的测试值。

### 41.4 数学和业务错误

把 attention softmax 放在 query 轴而不是 key 轴、把 mean 写成 max、对错误 token 区域做 pooling，pattern 都可能合法。

因此正确定位是：einops 扩大了可检查范围，并没有把动态 Python 张量代码变成完备静态类型系统。

## 42. “Stringly Typed”批评

最常见的批评是：pattern 是字符串，IDE 重构、静态类型检查和自动补全能力弱。这个批评成立一部分，但需要和位置元组 API 做现实比较。

```python
x.permute(0, 2, 3, 1)
```

这里的 `(0, 2, 3, 1)` 同样是一组缺少静态语义的字面量。类型检查器最多知道它们是 integers，无法知道 `2` 是 height。pattern 虽是字符串，却把轴语义和两侧结构暴露给运行时 parser，也让 reviewer 能读出意图。

更准确的比较是：

| 方案 | 局部语义 | 运行时结构检查 | 静态重构 | 持久轴类型 |
| --- | --- | --- | --- | --- |
| `permute(0,2,3,1)` | 弱 | 主要检查编号范围 | 弱 | 无 |
| einops pattern | 强 | 中到强 | 弱到中 | 无 |
| named tensor | 强 | 强 | 取决于框架 | 有 |
| shape type system | 强 | 可选 | 强 | 函数边界/类型层 |

字符串 DSL 的风险不应被否认：复杂 pattern 可能拼写错误，跨文件重命名困难，通用 Python 类型工具通常不解析其 grammar。工程上应通过短 pattern、轴命名规范、测试和 lint/review 约束缓解。

## 43. 与 shape type system 的边界

jaxtyping、torchtyping 等工具通常在函数边界声明 shape：

```python
# 概念示意
def forward(x: Float[Tensor, 'batch tokens channels']) -> Float[Tensor, 'batch classes']:
    ...
```

它们回答“这个函数接受和返回什么结构”；einops 回答“函数内部这一步如何从输入结构得到输出结构”。两者可以互补：

```python
def split_heads(x, heads: int):
    return rearrange(
        x,
        'batch tokens (heads dim) -> batch heads tokens dim',
        heads=heads,
    )
```

类型系统可验证调用边界，einops 在运行时验证 `channels % heads == 0` 并执行变换。但动态长度、值依赖 shape、编译器符号维度和跨框架类型插件仍会限制静态证明能力。

不要把二者写成竞争替代关系：持久 shape tracking 与局部结构 DSL 解决的是不同层次的问题。

## 44. 性能结论的正确读法

论文 Table 1 比较了 attention、Vision Permutator 和 Glow unsqueeze 的原始 PyTorch 与 einops 版本，在 CPU/CUDA、有无 JIT、不同输入大小下的耗时。

![Einops performance comparison](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-performance-table.png)

*Source: Rogozhnikov, ICLR 2022, Table 1 via OpenReview archival copy.*

表格支持三个谨慎结论：

1. 在这些案例和当时版本上，einops-rich 实现与原生实现处于相近速度范围。
2. 小 GPU 输入上，Python/额外操作开销相对更可见。
3. attention 等案例的主要成本是矩阵乘法，rearrangement 往往不是总耗时主项。

它不支持以下强结论：

- einops 永远与手写最优 kernel 一样快。
- einops 会自动 fusion。
- 2022 年 PyTorch 1.7.1、CUDA 11.0、einops 0.3.2 的数字代表 2026 年框架。
- 任意 pattern 都只产生一次 view。

在现代 PyTorch/JAX 中，编译器可能融合、消除或重新安排底层操作，也可能因 dynamic Python 或 stride 产生 graph break/copy。性能必须在目标模型、shape、后端版本和硬件上复测。

## 45. 论文证据强度与可读性小实验

这篇论文不是典型模型论文，没有 ImageNet 精度表，也没有证明 einops 能提高模型指标。其证据由四部分组成：

1. 结构错误案例：Figure 1 展示合法 reshape 如何破坏元素结构。
2. 真实代码案例：Vision Permutator、Glow、多头注意力。
3. 性能与缓存：证明 abstraction overhead 在所测场景中可控。
4. 适用性统计：从流行 PyTorch 仓库抽取 16 类 reshape/transpose 片段进行改写。

论文还做了一个 8 人的可读性问卷。参与者至少有一年 tensor programming 经验，在没有函数说明和上下文的情况下推断若干 notation 的输出。部分基础问题全部答对；复杂 composite axes 仍造成初始困惑。作者也明确承认，小样本问卷无法捕捉大型代码库和长期维护中的真实价值。

因此，最合理的证据评级是：**强设计案例 + 有限用户观察 + 小规模性能验证 + 后续真实采用**。它足以支持工具设计的实用性，但不足以做严格的人因工程因果结论。

## 46. OpenReview 审稿争议

OpenReview 上的争议集中在“这是否足够像研究论文”。批评意见认为文章接近技术博客或库说明，缺少大规模用户研究、形式化理论和传统实验。支持意见则强调：张量结构错误广泛存在，notation 设计具有跨框架价值，论文提供了可运行实现、真实案例和清晰的工程影响。

这类争议反映工具论文的评价难点。一个 API 的价值可能表现为：

- 未来数年被大量模型代码采用。
- 降低 review 和修改成本。
- 让同一算法更容易跨框架迁移。
- 把某类 silent bug 变成局部异常。

这些收益很难在一次短期 benchmark 中完整测量。反过来，真实采用量也不能自动证明 notation 的所有设计都是最优。论文最终以 Oral 录用，说明评审体系认可“编程接口本身可以构成研究贡献”，但这不免除对证据边界的批判。

## 47. 论文之后的版本演进

按官方 README、release 与源码，关键演进可整理为：

| 版本 | 主要变化 | 与论文的关系 |
| --- | --- | --- |
| 论文时期 | `rearrange / reduce / repeat`、layers、EinMix 基础 | ICLR 2022 主线 |
| `0.5` | `einops.einsum` | 扩展到 contraction |
| `0.6` | `pack / unpack` | 处理可变 rank 打包与可逆元数据 |
| `0.6.1` | Paddle backend | 扩展框架覆盖 |
| `0.7` | Array API、简化 `torch.compile` | 适配标准与编译生态 |
| `0.8.0` | tinygrad backend | 扩展轻量后端 |
| `0.8.1` | wheel 中分发 tests | 改善后端与安装验证 |
| `0.8.2` | 完整 MLX backend、Python 3.9+ | 本文稳定基线 |
| `0.9.0.dev0` | 预发布开发线 | 不作为稳定承诺 |

版本演进呈现了一个清晰策略：保留小型核心 notation，在周边增加 contraction、packing、标准接口和 backend。它没有演化成包含卷积、采样、索引、稀疏图和分布式通信的万能 DSL。

## 48. 作者五周年复盘

作者在五周年复盘中讨论了若干比 API 教程更有价值的维护经验。

### 48.1 零核心依赖

Einops 必须能被只使用任意一个框架的项目安装。若核心包强依赖 NumPy、PyTorch 或 TensorFlow，就会破坏其轻量跨框架定位。可选 backend 只有在用户已经导入对应框架后才参与识别。

### 48.2 接口克制

用户会提出许多新操作请求，但每加入一个 operation，都要为所有 backend、symbolic shape、JIT、错误信息、文档和测试承担长期成本。拒绝功能也是维护设计的一部分。

### 48.3 Symbolic shape 比整数 shape 难

很多看似简单的整除和 product 逻辑，在 TensorFlow graph、JAX tracing、Torch SymInt 下都不能直接转成 Python bool。通用库需要避免过早 concretize。

### 48.4 View 语义不能统一

用户希望“可读、跨框架、零拷贝”同时成立，但 storage model 是 backend 的属性。库能表达意图和选择合理原语，不能覆盖所有布局差异。

这篇复盘补充了论文未完全展开的现实：一个成功的小型 DSL，主要成本不是 parser 本身，而是多年维持兼容边界。

## 49. 与原生 NumPy/PyTorch API 的比较

Einops 并非每一行都优于原生 API。简单操作应按团队可读性选择。

### 适合保留原生写法

```python
x = x.T
x = x.squeeze(-1)
x = x.mean(dim=-1)
```

这些单步操作意图明确，额外 pattern 未必增加信息。

### 适合改写为 einops

```python
x = x.view(b, h, w, heads, dim)
x = x.permute(0, 3, 1, 2, 4)
x = x.contiguous().view(b, heads, h * w, dim)
```

可改为：

```python
x = rearrange(
    x,
    'b h w (heads dim) -> b heads (h w) dim',
    heads=heads,
)
```

选择标准不是字符数，而是是否存在多步 shape choreography、多个同长度轴、复合轴或可逆关系。

## 50. 与 Named Tensor、xarray 的比较

Einops 的轴名绑定在**操作 pattern** 上；xarray 或 named tensor 的轴名绑定在**tensor object** 上。

持久 named axes 的优势是跨操作保留标签、按名字对齐、减少全程序范围的位置歧义；代价是需要数据结构和算子生态全程理解标签，遇到不支持的 kernel 或框架边界时可能丢失元数据。

Einops 更轻：普通 tensor 不变，只在结构变换处命名轴。它容易接入现有模型，也意味着每个调用者都可能用不同名称，跨操作一致性靠团队规范。

两者不是严格互斥。xarray 数据进入神经网络前可转换为框架 tensor，内部用 einops 表达结构；named tensor 生态也可在需要 composite axis 时借鉴 pattern 思想。但必须明确标签在哪一层存在，避免以为 einops 调用后 tensor 自动记住了轴名。

## 51. 与 einx 的关系

后来的 einx 尝试把 Einstein-inspired notation 扩展到更广泛的 elementwise、indexing、reduction、vmap 和通用 vectorization。其 ICLR 2026 论文可以看作对“数组操作 DSL 还能覆盖多大范围”的进一步探索。

Einops 与 einx 的边界大致是：

- einops 刻意维持少量高频结构操作，语法和执行链相对可预测。
- einx 追求更统一的 vectorized operation 表达力，代价是 grammar 与编译语义更丰富。

不能把后来的扩展倒推成 einops 原论文错误。小操作集是 einops 可维护性和易学性的组成部分；更强表达力则服务另一类用户。选择应根据团队要解决的是 shape choreography，还是更广泛的数组计算抽象。

## 52. 工程采用与迁移指南

一次性把全仓库所有 `reshape` 替换为 einops，通常收益不高且 review 风险大。更稳妥的迁移顺序如下。

### 52.1 建立轴命名规范

建议优先使用完整单词：

| 语义 | 推荐名称 |
| --- | --- |
| batch | `batch` 或全仓统一 `b` |
| sequence/token | `tokens`, `query`, `key` |
| image | `height`, `width`, `channel` |
| attention | `head`, `dim` |
| patch | `patch_h`, `patch_w` |
| time/view | `time`, `view` |

短名适合数学密集的小函数；跨团队公共模块更适合完整单词。最糟糕的是同一文件中 `n` 一会儿表示 tokens，一会儿表示 heads。

### 52.2 优先迁移高风险链

优先级从高到低：

1. 三步以上 reshape/permute/view 链。
2. attention heads split/merge。
3. patchify/unpatchify、pixel shuffle、space-to-depth。
4. 需要手写逆变换的代码。
5. 多模态 token concat/split。
6. 单步且清楚的 squeeze/mean 最后考虑。

### 52.3 保持行为测试

迁移时用固定随机种子和非对称 shape，比对原实现与 einops 实现的完整数值：

```python
x = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
expected = old_impl(x)
actual = new_impl(x)
torch.testing.assert_close(actual, expected)
```

使用 `2,3,5,7` 这类互不相同的轴长度，比全部用 8 更容易发现交换。

### 52.4 用 round-trip 验证可逆变换

```python
restored = unpatchify(patchify(x))
torch.testing.assert_close(restored, x)
```

### 52.5 性能热点单独 profile

迁移前后比较 wall time、峰值内存、contiguous/copy、编译图数量和 kernel trace。不要把代码可读性改动与性能结论捆绑。

## 53. Code Review Checklist

评审 einops 代码时，可以逐项检查：

1. 输入 shape 在调用点或函数契约中是否明确？
2. 轴名是否表达真实业务语义，而非只换成 `a b c d`？
3. composite axis 的内部顺序是否符合 C-order？
4. 分解轴的长度来源是否唯一、稳定且有检查？
5. `(h w)` 与 `(w h)` 是否经过数值测试？
6. operation 是否正确：消轴用 `reduce`，增轴用 `repeat`？
7. ellipsis 是否真的需要，还是掩盖了 rank 假设？
8. 正反变换是否可通过交换 pattern 表达并做 round-trip？
9. 输入可能非 contiguous 时是否检查 copy 和后续 kernel？
10. `pack/unpack` 的 `packed_shapes` 是否与同一次前向绑定？
11. `einsum` 的被消去轴是否符合数学公式？
12. `torch.compile`、JIT 或目标 backend 是否有覆盖测试？
13. 错误路径是否测试 rank、整除和缺失长度？
14. 简单原生操作是否被不必要地复杂化？

一个好 pattern 应让 reviewer 更快理解结构。如果需要长段注释解释每个字符，可能应该拆成两步或改用更完整轴名。

## 54. 复现清单、局限性与最终判断

### 54.1 推荐源码阅读顺序

1. [论文 OpenReview 页面](https://openreview.net/forum?id=oapKSVM2bcj) 的 Sections 1、3、4、6、7。
2. [官方基础教程](https://einops.rocks/1-einops-basics/) 建立 pattern 直觉。
3. [pack/unpack 教程](https://einops.rocks/4-pack-and-unpack/) 理解后论文 API。
4. [EinMix 教程](https://einops.rocks/3-einmix-layer/) 理解参数轴。
5. [`parsing.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/parsing.py) 看 grammar。
6. [`einops.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py) 看 recipe 和执行。
7. [`_backends.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/_backends.py) 看 dispatch。
8. [`packing.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/packing.py) 看 wildcard metadata。
9. [`_torch_specific.py`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/_torch_specific.py) 看编译特殊路径。
10. [`einops/tests`](https://github.com/arogozhnikov/einops/tree/8e911db71f2e693a0c434b041180388c685ed06f/einops/tests) 看跨 backend 与错误契约。

### 54.2 局限性

Einops 的边界必须明确：

- 字符串 DSL 的静态分析和重构能力有限。
- 轴名只存在于单次操作，不做跨操作 shape tracking。
- C-order 和 composite axis 顺序仍需要程序员正确理解。
- backend 的 view/copy、symbolic shape、dtype 和编译差异会泄漏。
- 复杂 pattern 也可能变得难读，尤其是多层括号、数字轴和 ellipsis 混用时。
- 它不是 kernel compiler，不保证 fusion、加速或零拷贝。
- 它不是完整数组编程语言，也不覆盖任意 indexing、scatter/gather、稀疏或分布式语义。
- 它不能发现业务轴命名错误和数学目标错误。

### 54.3 最终判断

Einops 的长期价值，不在于把 `permute` 从 15 个字符缩成 12 个字符，而在于建立了一种跨团队、跨框架可共享的张量结构语言。它让输入、输出和轴关系在同一行出现；让 rank、轴集合、整除和新增长度变成可执行检查；又通过 recipe compiler 和 backend adapter 把这套语义落回框架原生操作。

它最成功的设计也正是它的克制：核心 pattern 足够小，可以被读懂、缓存、验证并稳定适配多个后端。后续的 `einsum`、`pack/unpack`、Array API、`torch.compile` 和 MLX 扩展了边界，但没有改变核心判断：**shape 是程序语义，不应只存在于注释、位置编号和程序员短期记忆里。**

使用 einops 时同样需要克制。不要把可读 notation 当成静态类型系统，不要把后端原语适配当成 kernel 优化器，也不要把“可能返回 view”当成零拷贝 SLA。把它用于真正复杂、容易漂移的结构变换，并配合非对称数值测试、round-trip、shape types 和 profiler，才能兑现论文所说的 clear and reliable。

## 参考资料

- Alex Rogozhnikov, [Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation](https://openreview.net/forum?id=oapKSVM2bcj), ICLR 2022.
- [ICLR 2022 Oral page](https://iclr.cc/virtual/2022/oral/6603) and [Poster page](https://iclr.cc/virtual/2022/poster/6602).
- [Einops official documentation](https://einops.rocks/).
- [Einops official GitHub repository](https://github.com/arogozhnikov/einops).
- [Einops v0.8.2 release](https://github.com/arogozhnikov/einops/releases/tag/v0.8.2).
- [Einops 0.8.2 on PyPI](https://pypi.org/project/einops/0.8.2/).
- Alex Rogozhnikov, [Retrospective thoughts on einops](https://arogozhnikov.github.io/2023/07/13/retrospective-thoughts-on-einops.html), 2023.
- [Python Array API standard](https://data-apis.org/array-api/latest/).
- [einx: Universal Tensor Operations in Einstein-Inspired Notation](https://openreview.net/forum?id=QqvQ3iAdpC), ICLR 2026.

## 附录 A：Pattern 逐例推导

这一附录不再增加新 API，而是把最容易“看懂结果、没看懂索引”的几个 pattern 展开。读者可以把它当作 review 时的手算模板。

### A.1 从 NCHW 到 patch tokens

输入：

```text
x.shape = (batch=2, channel=3, height=6, width=10)
patch_h = 2
patch_w = 5
```

Pattern：

```python
patches = rearrange(
    x,
    'batch channel (grid_h patch_h) (grid_w patch_w) '
    '-> batch (grid_h grid_w) (patch_h patch_w channel)',
    patch_h=2,
    patch_w=5,
)
```

第一步由运行时 shape 推断：

$$
grid_h=6/2=3,
\qquad
grid_w=10/5=2.
$$

输出 token 数为 $3\times2=6$，每个 token feature 为 $2\times5\times3=30$，所以输出 shape 是 `(2, 6, 30)`。

但 shape 只是第一层验证。元素顺序由右侧括号决定：token 轴 `(grid_h grid_w)` 中 `grid_w` 变化更快；feature 轴 `(patch_h patch_w channel)` 中 `channel` 变化最快。若下游线性层的训练数据按 channel-first patch flatten，pattern 应写成 `(channel patch_h patch_w)`，两者不可仅凭同样的 30 维 shape 互换。

### A.2 Space-to-depth

```python
y = rearrange(
    x,
    'b c (h block_h) (w block_w) -> b (c block_h block_w) h w',
    block_h=2,
    block_w=2,
)
```

若输入 `(1, 3, 8, 8)`，输出 `(1, 12, 4, 4)`。元素总数验证：

$$
1\cdot3\cdot8\cdot8
=1\cdot(3\cdot2\cdot2)\cdot4\cdot4.
$$

这里没有 reduction；空间局部位置被移入 channel composite axis。逆向 pattern 是：

```python
x2 = rearrange(
    y,
    'b (c block_h block_w) h w -> b c (h block_h) (w block_w)',
    block_h=2,
    block_w=2,
)
```

### A.3 按窗口做 max pooling

```python
y = reduce(
    x,
    'b c (h kh) (w kw) -> b c h w',
    'max',
    kh=2,
    kw=2,
)
```

执行语义不是先把整张图 flatten 后 max，而是先把输入轴拆为 `b,c,h,kh,w,kw`，重排使 `kh,kw` 成为 reduction axes，再对二者做 max，保留 `b,c,h,w`。

若输入高度不能被 `kh` 整除，einops 不会像某些 pooling kernel 那样隐式 padding 或丢弃边缘；它直接报 shape mismatch。padding 是另一个操作，应在调用前显式决定。

### A.4 按组归一化前拆 channel

```python
grouped = rearrange(x, 'b (groups channels) h w -> b groups channels h w', groups=32)
mean = reduce(grouped, 'b groups channels h w -> b groups 1 1 1', 'mean')
```

这段代码虽然展示了语义，但真实 GroupNorm 还需要 variance、epsilon、affine 参数和高效 fused kernel。Einops 适合表达结构，不意味着应以 Python 组合替代优化过的框架算子。此例的正确用途是解释、原型和测试，而不是声称性能等价。

### A.5 Beam search 状态复制

```python
# state: (batch, hidden)
beams = repeat(state, 'batch hidden -> batch beam hidden', beam=4)
```

与 `'batch hidden -> (batch beam) hidden'` 的差异是：前者保留显式 beam 轴，后者立刻把 batch 与 beam 合并。若后续 cache 或 logits 需要单独访问 beam，过早合并会使结构难追踪。Pattern 不只描述 shape，也鼓励在仍有语义价值时保留轴。

### A.6 时间与 batch 合并

```python
flat = rearrange(x, 'batch time channel -> (batch time) channel')
restored = rearrange(flat, '(batch time) channel -> batch time channel', batch=batch)
```

逆向至少要知道 `batch` 或 `time`。如果两者都是动态且未保存，就不能从乘积唯一恢复。这个例子说明可逆性不只取决于箭头两侧轴集合，还取决于运行时是否保留分解参数。

### A.7 Ellipsis 下的 channel-last 转换

```python
y = rearrange(x, '... channel -> ... channel')
```

这是 identity，通常没有价值。更实际的用法是固定末尾结构：

```python
# (..., heads, dim) -> (..., heads*dim)
y = rearrange(x, '... heads dim -> ... (heads dim)')
```

它可处理 `(batch, tokens, heads, dim)` 和 `(batch, views, tokens, heads, dim)`。代价是调用点看不到 ellipsis 里是什么。若模块只允许一种 rank，写完整轴名更能捕获错误。

### A.8 原论文 Listing 1 的统一性

论文 Listing 1 把 NumPy 中 transpose、reshape、squeeze、expand_dims、stack、concatenate、flatten、split、max、mean、repeat 和 tile 的常见形式放在一张对应表中。

![Correspondence between mainstream operations and einops](/images/blog/einops-clear-reliable-tensor-manipulations/einops-paper-mainstream-operations.png)

*Source: Rogozhnikov, ICLR 2022, Listing 1 excerpt via OpenReview archival copy.*

这张表的意义不是证明所有 NumPy API 都应废弃，而是证明三类轴关系可以覆盖大量高频结构操作。einops 的统一发生在“轴如何从输入对应到输出”，不是在数组计算的所有维度上统一。

## 附录 B：`pack`、`einsum` 与 backend 的源码细节

### B.1 `pack()` 如何处理不同 rank

[`pack()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/packing.py#L33-L94) 首先由 `analyze_pattern()` 找到 wildcard 前后固定轴的数量。例如 `'batch * channel'` 表示 wildcard 前 1 个轴、后 1 个轴。

对每个输入 tensor，它执行：

1. 检查 rank 至少包含固定轴。
2. 取 wildcard 覆盖的原始 shape，追加到 `packed_shapes`。
3. 将 wildcard 覆盖的所有轴 reshape 为单个 `-1`。
4. 沿 wildcard 所在位置 concat。

例子：

```text
(2, 3, 5)       -> wildcard shape (3,)    -> (2, 3, 5)
(2, 4, 7, 5)    -> wildcard shape (4, 7)  -> (2, 28, 5)
(2, 6, 1, 2, 5) -> wildcard shape (6,1,2) -> (2, 12, 5)
concat result: (2, 43, 5)
```

公共 `batch=2` 和 `channel=5` 由 backend concat 自然验证；不一致时后端会报 shape 错误。

### B.2 `unpack()` 如何恢复切片

[`unpack()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/packing.py#L97-L169) 把每个 packed shape 的元素乘积转换为切片长度，再计算累积 split positions。它允许至多一个 packed shape 包含 `-1`；若有一个未知长度，前后已知段分别从左、右累加，中间剩余长度归给未知段。

切片后，每段 reshape 回：

```text
fixed prefix axes + packed original axes + fixed suffix axes
```

因此 `packed_shapes` 是运行时协议的一部分。把它序列化、跨 batch 复用或与另一条输入混用，可能产生合法但错误的恢复结果。最稳妥做法是让它与 packed tensor 同生命周期传递。

### B.3 `einsum` 如何把长轴名压缩为后端字符串

NumPy 风格 einsum 通常只接受单字符 label。[`_compactify_pattern_for_einsum()`](https://github.com/arogozhnikov/einops/blob/8e911db71f2e693a0c434b041180388c685ed06f/einops/einops.py#L788-L846) 会把多字符轴名映射到 `ascii_letters`：

```text
batch query head dim, batch key head dim -> batch head query key
```

可能被压缩为类似：

```text
abcd,aecd->acbe
```

具体字符不承载语义，只要同名轴映射一致。压缩结果再交给 backend 的 einsum。

源码显式要求 `->`，避免依赖 NumPy 的隐式输出轴排序；右侧出现未在输入中定义的轴会报错；超过可用单字符数量会失败。Composite axes、anonymous axes 和 singleton axes 当前抛出 `NotImplementedError`，这与 `rearrange` grammar 的能力边界不同。

### B.4 Einsum 的四种典型语义

矩阵乘法：

```python
y = einsum(a, b, 'row inner, inner col -> row col')
```

Batch 矩阵乘法：

```python
y = einsum(a, b, 'batch row inner, batch inner col -> batch row col')
```

Trace：

```python
trace = einsum(x, 'index index ->')
```

带任意 batch rank 的线性映射：

```python
y = einsum(weight, x, 'out_dim in_dim, ... in_dim -> ... out_dim')
```

重复轴在同一输入中可取 diagonal，这正是 `ParsedExpression(..., allow_duplicates=True)` 在 einsum 路径存在的原因。不要把 `rearrange` 禁止重复轴的规则套到 einsum。

### B.5 Backend 最小契约

`AbstractBackend` 的价值可以用“recipe 需要什么”来理解，而不是按框架逐个背 API：

| 能力 | recipe 用途 |
| --- | --- |
| `shape` | 重建 elementary axis 长度 |
| `reshape` | 拆/合 composite axes |
| `transpose` | elementary axes 重排 |
| `reduce` | 消去轴 |
| `add_axes` | repeat 新轴 |
| `stack/concat` | list rearrange 与 pack |
| `einsum` | contraction |
| `to_numpy/from_numpy` | 测试和互操作 |

一个新 backend 若能可靠提供这些原语，就可以接入大部分 einops 功能。难点不在写几十行 wrapper，而在 symbolic shape、JIT、布尔 reduction、zero-size axis、dtype、device 和 view/copy 的边缘行为。

### B.6 为什么 backend 测试以 NumPy 为参考

稳定版 README 说明，从 0.8.1 起 tests 随包分发，可运行指定 backend 的测试。测试通常用 NumPy 生成参考结果，再比较其他 backend。这种方式验证数值与 shape 一致性，但仍不能验证所有编译器模式和设备组合。

测试目录按问题拆分：

- `test_ops.py`：核心 rearrange/reduce/repeat。
- `test_parsing.py`：grammar 与非法表达式。
- `test_packing.py`：pack/unpack round trip。
- `test_einsum.py`：多 tensor contraction 与语法限制。
- `test_layers.py`：框架 layer。
- `test_examples.py`：教程和典型 pattern。
- `test_other.py`：backend、parse_shape 等补充路径。

源码阅读时，测试比文档更能说明错误契约。尤其是 parser 边界，不应只从正常示例推断。

## 附录 C：可执行验证矩阵

以下验证策略适合项目引入 einops 时直接采用。它不要求复刻论文 benchmark，而是针对结构错误设计。

### C.1 基础数值一致性

```python
import numpy as np
from einops import rearrange, reduce, repeat

x = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

np.testing.assert_array_equal(
    rearrange(x, 'b c h w -> b h w c'),
    x.transpose(0, 2, 3, 1),
)

np.testing.assert_array_equal(
    reduce(x, 'b c h w -> b c', 'sum'),
    x.sum(axis=(2, 3)),
)

np.testing.assert_array_equal(
    repeat(x, 'b c h w -> b copies c h w', copies=2),
    np.repeat(x[:, None], 2, axis=1),
)
```

选择互不相同的轴长度和 `arange`，可以让元素位置错误直接表现为数值差异。

### C.2 Patch round trip

```python
images = np.arange(2 * 3 * 6 * 10).reshape(2, 3, 6, 10)

patches = rearrange(
    images,
    'b c (h ph) (w pw) -> b (h w) (ph pw c)',
    ph=2,
    pw=5,
)
restored = rearrange(
    patches,
    'b (h w) (ph pw c) -> b c (h ph) (w pw)',
    h=3,
    w=2,
    ph=2,
    pw=5,
    c=3,
)

np.testing.assert_array_equal(restored, images)
```

### C.3 Attention heads round trip

```python
x = np.arange(2 * 11 * 30).reshape(2, 11, 30)
heads = rearrange(x, 'b tokens (head dim) -> b head tokens dim', head=5)
restored = rearrange(heads, 'b head tokens dim -> b tokens (head dim)')
np.testing.assert_array_equal(restored, x)
```

`tokens=11`、`head=5`、`dim=6` 都不同，能有效识别误交换。

### C.4 Pack/unpack 不同 rank

```python
inputs = [
    np.zeros((2, 5)),
    np.zeros((2, 3, 5)),
    np.zeros((2, 4, 7, 5)),
]
packed, ps = pack(inputs, 'b * c')
outputs = unpack(packed, ps, 'b * c')

assert [x.shape for x in outputs] == [x.shape for x in inputs]
for before, after in zip(inputs, outputs):
    np.testing.assert_array_equal(before, after)
```

### C.5 Einsum 交叉验证

```python
q = np.random.default_rng(0).normal(size=(2, 5, 3, 7))
k = np.random.default_rng(1).normal(size=(2, 11, 3, 7))

actual = einsum(
    q,
    k,
    'batch query head dim, batch key head dim -> batch head query key',
)
expected = np.einsum('bqhd,bkhd->bhqk', q, k)
np.testing.assert_allclose(actual, expected)
```

### C.6 错误路径必须真的报错

```python
import pytest
from einops import EinopsError

with pytest.raises(EinopsError):
    rearrange(np.zeros((2, 3, 5)), 'b c h w -> b h w c')

with pytest.raises(EinopsError):
    rearrange(np.zeros((2, 3, 5, 7)), 'b c h w -> b h w')

with pytest.raises(EinopsError):
    rearrange(np.zeros((2, 197, 8)), 'b (h w) c -> b h w c', h=14)

with pytest.raises(EinopsError):
    repeat(np.zeros((5, 7)), 'h w -> batch h w')
```

错误消息文本可能随版本调整，测试应优先断言异常类型和关键语义，而不是锁死整段字符串。

### C.7 后端矩阵

如果项目支持多个 backend，至少覆盖：

| 维度 | 建议样例 |
| --- | --- |
| dtype | float32、float16/bfloat16、int、bool reduction |
| layout | contiguous、transpose 后非 contiguous、slice |
| shape | 普通整数、zero-size axis、dynamic/symbolic |
| device | CPU、CUDA/MPS/MLX 中项目真实使用项 |
| mode | eager、JIT/script、`torch.compile` 或 JAX jit |
| operation | rearrange、reduce、repeat、pack、einsum |

不要为了形式完整测试未支持的框架；矩阵应对应真实部署路径。

### C.8 性能基准应该测什么

一个有意义的基准至少同时记录：

1. eager wall time，带 warm-up。
2. 编译时间和 steady-state time 分开。
3. kernel 数和 graph break。
4. 峰值显存/内存。
5. 输出 stride、contiguous 状态和是否发生 copy。
6. 真实 batch/sequence/resolution 分布，而非单一 shape。

若 einops 版本变慢，应先检查 pattern 是否导致不同的内存顺序，而不是直接归因于字符串解析。两级缓存命中后，解析通常不是大 tensor 的主成本。

### C.9 生产事故排查顺序

当迁移后指标异常但代码不报错，按以下顺序排查：

1. 用 `arange` 小 tensor 对比旧实现完整数值。
2. 打印每一步 shape 与 stride，不只最终 shape。
3. 检查 composite axis 内部顺序。
4. 检查 softmax/reduction 实际作用轴。
5. 检查同长度轴是否被交换。
6. 检查 `pack` 的 metadata 是否跨请求复用。
7. 检查 channels-first/channels-last 数据约定。
8. 最后再看 backend/compiler 差异。

这套顺序体现本文的核心判断：einops 能减少很多结构错误，但只要轴名声明本身错误，合法 pattern 仍会忠实执行错误意图。
