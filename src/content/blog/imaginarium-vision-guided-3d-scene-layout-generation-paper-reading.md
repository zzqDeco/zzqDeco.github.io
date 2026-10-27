---
title: "Imaginarium 论文精读：视觉引导、高质量资产库、代码实现与 3D 场景布局生成"
description: "从论文与官方代码双线精读 Imaginarium 如何通过 Flux 图像引导、视觉解析、资产检索、姿态估计、场景图约束和 Blender 物理优化生成高质量 3D 场景布局"
pubDate: "2026-06-26T10:11:20+08:00"
updatedDate: "2026-06-26T10:11:20+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "3D Scene Synthesis"
  - "Computer Vision"
  - "Generative AI"
  - "Graphics"
  - "Code Reading"
draft: false
---

[Imaginarium: Vision-guided High-Quality 3D Scene Layout Generation](https://arxiv.org/abs/2510.15564) 由 Xiaoming Zhu、Xu Huang、Qinghongbing Xie、Zhi Deng、Junsheng Yu、Yirui Guan、Zhongyuan Liu、Lin Zhu、Qijun Zhao、Ligang Liu、Long Zeng 撰写（arXiv 提交日期 2025 年 10 月 17 日，DOI `10.48550/arXiv.2510.15564`）。该工作发表于 **SIGGRAPH Asia 2025 & ACM Transactions on Graphics (TOG)**，官方提供了[项目主页](https://ydove0324.github.io/Imaginarium/)、[开源代码仓库](https://github.com/HiHiAllen/Imaginarium)（基于 `main` 分支）、[Hugging Face 核心数据集](https://huggingface.co/datasets/HiHiAllen/Imaginarium-Dataset)以及[预计算派生数据集](https://huggingface.co/datasets/binicey/Imaginarium-3D-Derived-Dataset)。

Imaginarium 的核心思路是将 **2D 图像生成模型的审美、构图与多物体组合先验** 转译为 **可复用、可编辑的 3D 资产布局**：

```text
文本提示
  -> 微调 Flux 生成 2D guide image
  -> VLM / 检测 / 分割 / 深度估计解析图像
  -> 从高质量资产库检索 3D asset
  -> 估计旋转、平移、缩放
  -> 用 scene graph、优化和 Blender physics 修正布局
  -> 输出可渲染的 3D scene layout
```

下文结合原论文与官方 `main` 分支源码，从问题定义、全局架构与中间文件契约、Flux 引导图像生成、S0–S4 五阶段核心算法、实验消融到工程复现链路进行系统拆解。

## 1. 任务边界与问题形式化：以 2D 视觉先验驱动 3D 资产布局

### 1.1 四层决策架构与核心概念边界

面向游戏场景搭建、CGI 制作、虚拟拍摄与室内外设计等工业场景，3D 场景布局生成（3D scene layout generation）需要输出由独立资产、空间变换、支撑层级与物理约束共同构成的可交互场景。Imaginarium 将这一目标拆解为四个递进层级：

| 层级 | 问题 | Imaginarium 的做法 |
| --- | --- | --- |
| 内容层 | 这个场景应该有什么物体 | 用文本提示和 2D 图像生成扩展场景内容 |
| 资产层 | 用哪个 3D 模型表示物体 | 从预定义高质量 asset library 检索 |
| 几何层 | 资产如何摆放 | 从图像分割、深度、OBB、姿态匹配估计变换 |
| 逻辑层 | 布局是否合理 | 用 scene graph、support tree、wall constraints、physics 修正 |

由于 3D 场景布局真值数据稀缺，直接训练端到端 3D 布局生成器的泛化能力受限；而大规模 2D 图像生成模型已经具备成熟的构图、光影、材质与局部细节密度先验。Imaginarium 因此将 2D 引导图像（guide image）作为核心桥梁：先由微调后的图像生成模型产出构图合理、风格统一的参考画面，再通过视觉解析、资产检索与几何物理优化将其还原至 3D 资产空间。

理解这套管线需明确以下核心概念的任务边界：
- **预定义资产库（Predefined asset library）与资产检索（Asset retrieval）**：系统在包含网格、材质、尺寸、类别、多视角渲染图、特征嵌入及容器内部可放置空间的有限资产库中选择模型。检索过程同时考量类别匹配、视觉特征相似度、尺寸兼容性与成套家具风格一致性。
- **姿态估计（Pose estimation）与带朝向包围盒（OBB）**：给定生成图像与候选 3D 资产，利用单目深度与实例掩码提取前景点云并拟合 OBB，结合多视角模板匹配估计资产在场景中的三维旋转、平移与尺度。
- **场景图（Scene graph）与物理合理性（Physical plausibility）**：场景图将物体间的支撑层级、靠墙/吊挂属性、朝向与成套分组抽象为可执行约束；物理合理性则要求消除无支撑悬空与严重穿模，并对抱枕、堆叠书本等对象执行刚体物理仿真。

### 1.2 数学形式化与两阶段代码映射

论文将场景生成形式化为给定文本提示 $prompt$ 与预定义资产库 $A$ 下，到场景对象集合 $O$ 的映射：

$$
G(O \mid prompt, A)=\{o_1,o_2,o_3,\dots,o_n\}, \quad o_i=\{\text{obj}_i,R_i,t_i,s_i\}.
$$

各符号的数学定义与源码载体对应如下：

| 符号 | 含义 | 代码侧对应 |
| --- | --- | --- |
| $\text{obj}_i$ | 从资产库选中的 3D asset | `retrieval_results_final.json` 中的 asset id/name |
| $R_i$ | asset 旋转矩阵或姿态 | `S3_pose_inference` 生成的 pose matrix |
| $t_i$ | 3D translation | `placement_info.json` 中的位置字段 |
| $s_i$ | scale | S3/S4 根据 OBB、资产尺寸、容器空间修正 |
| $A$ | asset library | `asset_data/imaginarium_assets/` 与 `imaginarium_asset_info.csv` |
| $O$ | 输出 object set | S4 后的 refined placement info 与 Blender scene |

这一形式化背后包含两重挑战：其一，$prompt \to O$ 是典型的一对多映射（一句 “cozy living room” 对应无数种合法布局），系统通过 2D 引导图像将开放文本收敛为单一确定的视觉构图；其二，$\text{obj}_i$ 必须严格取自有限资产库 $A$，若 2D 图像生成了资产库中不存在的新奇混合拓扑（如开放书架与封闭衣柜的混合体），检索只能退而求其次选择近似资产，进而影响下游子物体的支撑放置。

在开源实现中，公式中的 $G$ 被拆分为两个独立的命令行入口：

$$
G(O \mid prompt,A)
=
\text{I2Layout}(\text{T2I}(prompt;A),A).
$$

其中 `run_imaginarium_T2I.py` 通过在 Flux 上挂载经资产库渲染图微调的 LoRA 权重生成 2D 引导图像，`run_imaginarium_I2Layout.py` 则调用 `ImaginariumPipeline.run()` 依次执行 `GeometryModule`、`SemanticParsingModule`、`RetrievalModule`、`PoseModule` 与 `LayoutModule` 五个阶段。

## 2. 全局管线与源码架构：从论文模块到 S0–S4 文件契约

### 2.1 论文模块、代码阶段与中间文件契约总表

原论文 Fig. 2 按方法贡献将系统划分为提示扩展（Prompt Expander）、场景图像分析（Scene Image Analysis）、场景布局重建（Scene Layout Reconstruction）与布局优化（Layout Refinement）；而在开源工程中，为支持长耗时任务的断点恢复与中间结果审查，I2Layout 内部细分为 `S0` 至 `S4` 五个阶段。

下表汇总了论文模块、源码文件入口、核心函数、中间文件契约及常见失败信号的完整对应关系：

| 阶段 / 论文模块 | 源码入口与核心实现 | 关键中间文件或 `Context` 字段 | 下游消费方式与关键检查点 | 常见失败信号 |
| --- | --- | --- | --- | --- |
| **T2I Prompt Expander**<br>（Flux 引导图像生成） | `run_imaginarium_T2I.py`<br>`scripts/flux/train.py` | `generated_images/*.png` | 作为 I2Layout 的视觉输入；检查 1024 分辨率、视角稳定性与资产库风格契合度 | 物体严重遮挡、透视畸变、生成资产库缺失的新奇拓扑 |
| **S0 Geometry**<br>（单目深度与点云构建） | `modules/geometry.py`<br>（`GeometryModule.run()`、`estimate_focal_length()`、`create_point_cloud()`） | `S0_geometry_pred_results/depth.png`<br>`Context`: `depth_image`、`ori_image_numpy`、`intrinsics`、debug `pcd.ply` | S1 地面/墙面 RANSAC 拟合、S2 物体 OBB 估计、S3 深度对齐 | 深度尺度漂移、墙地交界模糊、固定焦距近似导致点云拉伸 |
| **S1 Semantic Parsing**<br>（VLM 语义、检测分割与场景图） | `modules/parsing.py`<br>`modules/_s1_legacy_functions.py`<br>`utils/ransac.py`<br>`prompts/used_prompts.py` | `S1_scene_parsing_results/` 下：<br>`masks.pkl`、`final_detect_items.pkl`、`floor_walls_pose.json`、`scene_graph_result_final.json` | S2 按 mask 裁剪图像检索资产；S3/S4 依据场景图执行支撑、靠墙、朝向与成套分组约束 | 漏检小物体、重复框未去重、RANSAC 墙地平面颠倒、VLM 父子支撑幻觉 |
| **S2 Asset Retrieval**<br>（OBB、尺寸修正与资产/贴图检索） | `modules/retrieval.py`<br>`modules/_s2_legacy_functions.py`<br>`utils/obb.py`（`TextureRetrieval`） | `S2_3d_retrieval_results/` 下：<br>`pcd_obb_data.json`、`retrieval_results_final.json`、`texture_retrieval_results.json` | S3 读取候选资产与 OBB 估计 6D/9D 姿态；S4 加载对应网格与背景墙地贴图 | 类别正确但外观/风格错配、单目点云导致 OBB 尺寸失真、成套椅子检索不一致 |
| **S3 Pose Estimation**<br>（162 视角匹配、单应性重排与几何增强） | `modules/pose.py`<br>`modules/_s3_legacy_functions.py`<br>`utils/view_matching.py`<br>`models/ae_net/` | `S3_pose_inference/` 下：<br>`{scene}_placement_info.json`、`pose_prediction_stitched.png` | S4 的初始旋转、平移、尺度与截断标记；通过拼接对比图审查模板视角与裁剪图是否对齐 | 对称物体 180° 翻转、边缘截断物体平移漂移、小物体 OBB 退化 |
| **S4 Layout Refinement**<br>（局部对齐、体素模拟退火与 Blender 物理） | `modules/layout.py`<br>`modules/S4_blender_layout_and_corr.py`<br>`scripts/drop_sim_script.py` | `S4_layout_refinement/` 下：<br>`{scene}_placement_info_s4.json`、`{scene}_render_simu.png` | 最终 3D 场景文件与渲染图；依赖预计算体素与容器内部可放置空间修正穿模与悬空 | 残留穿模、悬空、靠墙方向反向、刚体掉落后偏离视觉位置 |

> **贯穿全程的视觉引导**：图像信息在管线中被四次深度消费——T2I 生成引导图，S1 解析物体掩码与场景图，S2 基于物体裁剪图检索 3D 资产与背景贴图，S3 则通过裁剪图与 162 个渲染模板做局部特征匹配估计姿态。

### 2.2 调用栈、Context 数据总线与 Wrapper/Legacy 分层

从命令行到 Blender 子进程的完整调用链如下：

```text
run_imaginarium_I2Layout.py
  -> ImaginariumPipeline(...)
    -> Context(...)
    -> GeometryModule.run()
    -> SemanticParsingModule.run()
    -> RetrievalModule.run()
    -> PoseModule.run()
    -> LayoutModule.run()
      -> blender --background --python modules/S4_blender_layout_and_corr.py
```

源码在系统工程层面做出了四项关键设计：

1. **延迟导入（Lazy Import）与 `Context` 全局数据总线**：`pipeline.py` 移除了顶层重型库导入以加快启动速度；`core/context.py` 统一管理输出目录（`saved_results/{image_name}_result/`）、阶段日志（`pipeline.log` 与 `stage_logs/`）、中间内存字典（`set_data()` / `get_data()`）以及 DINOv2、AENet、VLM 客户端的按需加载与显存释放。
2. **断点恢复（Resume）、清理（`--clean`）与调试（`--debug`）模式**：默认情况下，各阶段检测到目标输出文件已存在即自动跳过，避免因网络波动或单阶段报错而重跑数分钟的前置流程；`--clean` 强制清空历史缓存从头执行；`--debug` 则额外导出点云、掩码叠加与姿态拼接图（如 `pose_prediction_stitched.png`），便于将失败原因精准定界到具体阶段。
3. **Wrapper 层与 Legacy 实现层分离**：`modules/parsing.py`、`modules/retrieval.py`、`modules/pose.py` 主要承担目录准备、缓存检查与 `Context` 读写；核心算法逻辑则封装在 `_s1_legacy_functions.py`、`_s2_legacy_functions.py` 与 `_s3_legacy_functions.py` 中。二次开发时可优先在 Wrapper 层插入校验与可视化钩子，无需侵入底层复杂函数。
4. **独立 Blender 运行时与仓库演进差异**：`LayoutModule.run()` 在启动前主动释放主进程 PyTorch 模型显存，随后通过子进程调用 Blender Python 环境执行网格加载、体素碰撞、模拟退火与刚体物理。此外，当前开源仓库相比原论文还包含多项后续工程增强：LLM/VLM 接口通过 `utils/llm_api.py` 与 `config-example.yaml` 改为可配置端点（支持 GPT-4o 或 Claude Sonnet 4.5 等模型），并新增了背景墙地贴图检索（Background Texture Support）与场景图成套物体分组（Scene Graph Groups）。

## 3. 文本扩展与高质量资产库：Flux 微调（T2I）与数据集构建

### 3.1 Flux LoRA 引导图像生成（Prompt Expander）

为了将简短的文本提示转化为兼具审美构图且贴合自有资产库风格的 2D 引导图像，论文在 `black-forest-labs/FLUX.1-dev` 的 Transformer 模块上引入 LoRA 适配器（秩 `r=16`，`lora_alpha=16`），使用资产库的多视角场景渲染图进行 DreamBooth/LoRA 微调。

`run_imaginarium_T2I.py` 的核心配置与推理逻辑如下：

| 参数 | 默认值或含义 | 解释 |
| --- | --- | --- |
| `--base_model` | `black-forest-labs/FLUX.1-dev` | 基础 Flux 模型 |
| `--lora_path` | `weights/imaginarium_finetuned_flux.pth` | 资产库风格微调权重 |
| `--guidance_scale` | `7.5` | 文本引导强度 |
| `--num_inference_steps` | `50` | 采样步数 |
| `--resolution` | `1024` | 输出图像分辨率 |
| `--num` | 生成张数 | 可一次生成多个候选 guide image |

```python
transformer = FluxTransformer2DModel.from_pretrained(...)
transformer.add_adapter(LoraConfig(r=16, lora_alpha=16, ...))
set_peft_model_state_dict(transformer, lora_state_dict)
pipeline = FluxPipeline.from_pretrained(..., transformer=transformer)
image = pipeline(prompt, height=1024, width=1024, ...).images[0]
```

将 T2I 独立于 I2Layout 之外具有明确的工程价值：用户既可以从单次生成的 4 张候选图中挑选构图最佳、遮挡最少的一张送入下游，也可以结合 Canny 边缘控制或局部重绘（inpainting）对引导图进行交互编辑后再触发 3D 重建。微调后的 Flux 显著缩小了生成图像与 3D 资产库之间的域间隙，从源头降低了下游检索错配与姿态估计失败的概率。

### 3.2 高质量资产库与预计算派生数据集

Imaginarium 的场景表现力很大程度上建立在其构建的高密度、专业级 3D 资产与场景数据集之上。由 20 位具有三年以上经验的专业 3D 艺术家使用定制模型、开源精品资产与商用授权资产搭建而成，其核心规模指标远超传统室内数据集 3D-FUTURE：

| 统计维度 | Imaginarium 数据集 | 3D-FUTURE 对比 |
| --- | --- | --- |
| 3D 场景资产总数（Scene Assets） | `2,037` | — |
| 专业手工设计场景布局（Layouts） | `147` | — |
| 细粒度资产类别（Asset Classes / Categories） | `500` 类 / `237` 大类 | `34` 类 |
| 覆盖场景类型（Scene Types） | 正文约 `20` 类（附录统计 `21` 类） | 集中于常规室内房型 |
| 单场景平均物体数量（Objects / Scene） | **31.86** | `5.09` |

在 `config/config-example.yaml` 中，系统运行依赖两大类数据包与预计算缓存：
- **基础数据包（`HiHiAllen/Imaginarium-Dataset`）**：包含推理用 3D 资产模型（`imaginarium_assets.tar.gz`）、资产元数据表（`imaginarium_asset_info.csv`）、容器内部可放置空间标注（`imaginarium_assets_internal_placement_space.tar.gz`）、背景墙地贴图库（`background_texture_dataset.tar.gz`）、Flux 微调训练集（`flux_train_data.tar.gz`）以及含 `.blend` 源文件、RGB、深度、包围盒与掩码的完整研究数据集（Part 1–4）。
- **派生预计算包（`binicey/Imaginarium-3D-Derived-Dataset`）**：包含每个资产在 162 个视角下的渲染模板图（`imaginarium_assets_render_results_part[1-4].tar.gz`）、DINOv2 图像块特征嵌入（`imaginarium_assets_patch_embedding.tar.gz`）、用于 S4 快速碰撞检测的预计算体素（`imaginarium_assets_voxels.tar.gz`），以及 Depth Anything V2、DINOv2 ViT-L/14、AENet 的预训练权重与预配置 Blender 4.3.2 运行包。

## 4. 几何与语义解析（S0–S1）：深度点云、平面拟合与场景图构建

### 4.1 S0 Geometry：单目度量深度与相机内参近似

`modules/geometry.py` 中的 `GeometryModule` 负责将 2D 引导图像提升至 3D 相机坐标系。系统加载 `depth_anything_v2_metric_hypersim_vitl.pth` 预测单目度量深度图 `depth.png`，并根据图像宽度采用固定画幅比例近似估计相机焦距：

```python
def estimate_focal_length(width):
    return int(30 / 36 * width)
```

该实现假设输入引导图近似对应 Unreal Engine 5 的标准相机设置（30mm 焦距、36mm 传感器画幅），令 $f_x=f_y=\frac{30}{36}W$、主点 $(c_x,c_y)=(W/2,H/2)$，进而通过 `create_point_cloud` 将像素反投影为三维点云。这一相机先验直接决定了后续 S1 墙地平面拟合、S2 物体 OBB 尺度以及 S3 平移初值的全局度量基准。

### 4.2 S1 Parsing：VLM 物体识别、Grounding-DINO + SAM 与 RANSAC 平面提取

`SemanticParsingModule.run()`（底层调用 `modules/_s1_legacy_functions.py` 中的 `run_scene_parsing_pipeline`）串联了视觉语言解析、开放词汇检测分割与背景几何重建：

```text
image + depth
  -> VLM object parsing
  -> Grounding-DINO detection
  -> SAM segmentation
  -> secondary inspection / dedup
  -> floor / wall / ceiling parsing
  -> RANSAC plane fitting
  -> scene graph construction
  -> masks.pkl / final_detect_items.pkl / scene_graph_result.json / floor_walls_pose.json
```

1. **受控词表 VLM 解析与检测分割**：首先由 VLM 在资产库预定义类别集合内识别图像中的物体清单，经过二次核查与去重后，将物体名称输入 Grounding-DINO 预测 2D 检测框，再由 SAM 提取像素级实例掩码 `masks.pkl`。受限于单图分辨率与遮挡，小尺寸装饰物的检测恢复难度明显高于主家具（论文 Table 3 中次要物体恢复率为 `70.41%`，主要物体达 `92.31%`）。
2. **RANSAC 背景墙地平面拟合**：`utils/ransac.py`（含 `fit_floor_pcd`、`fit_walls_pcd_and_pred_xyz`、`find_floor`、`find_walls`、`estimate_floor_and_walls`）结合背景掩码与 S0 点云，拟合地面、墙面与天花板平面方程并导出 `floor_walls_pose.json`，建立整个房间的世界坐标系基准。

### 4.3 VLM 提示工程与可执行场景图构建

Imaginarium 并未让 VLM 直接猜测三维坐标，而是通过 `prompts/used_prompts.py` 中的结构化提示词，引导 VLM 输出稳定的离散拓扑关系，再结合几何重叠度转化为 S4 优化器的硬约束：

- **三层层级遍历（`SCENE_HIERARCHICAL_TRAVERSAL_PROMPT`）**：要求 VLM 按“功能区域（area）$\to$ 父物体（parent object）$\to$ 子物体（child objects）”三级结构输出 JSON，且物体类别必须严格落在预定义资产类别表内，从源头将桌面书本、床上抱枕、柜内摆件归入支撑树（support tree）的子节点。
- **支撑、靠墙与吊挂判定（`GENERATE_SCENE_GRAPH_PROMPT_S1` & `FLOOR_WALL`）**：结合 `generate_scene_graph_geometry`、`verify_floor_parent_with_vlm`、`supported_generate` 与 `against_wall_generate_top_down` 等函数，为每个物体判定支撑父节点 `parent`（记作 $\text{obj}_a \prec \text{obj}_b$，附录评测地面支撑树准确率达 `91.95%`）、是否靠墙 `isAgainstWall`、是否落地 `isOnFloor`、是否天花板吊挂 `isHangingFromCeiling`（如吊灯）以及是否墙面挂载 `isHangingOnWall`（如挂画）。
- **成套分组与朝向关系（`GENERATE_SEMANTIC_RELATIONSHIPS_PROMPT`）**：调用 `analyze_groups_and_facing_relations` 识别桌椅围合的正面朝向（`directlyFacing`）以及视觉外观完全一致的成套家具组（Groups）。同组对象（如围绕餐桌的多把餐椅）在后续 S2 阶段将强制共享同一个检索到的 3D 资产，有效避免了同类物体各自检索导致风格杂乱的问题。

## 5. 资产检索与尺寸修正（S2）：DINOv2 匹配与背景贴图

`RetrievalModule.run()` 负责将 S1 切分出的每个前景实例掩码映射为资产库中的具体 3D 模型，并输出 `pcd_obb_data.json`、`retrieval_results_final.json` 与 `texture_retrieval_results.json`：

```text
S1 masks + depth
  -> compute OBB data (cal_pcd_obb_data)
  -> VLM refine dimensions (refine_dimensions_with_vlm)
  -> DINOv2 asset retrieval (visual similarity + size compatibility)
  -> background texture retrieval
```

1. **点云 OBB 估计与 VLM 尺寸修正**：`cal_pcd_obb_data` 先根据实例掩码与深度点云拟合初始 OBB。由于单目深度在物体边缘、遮挡区或薄壁结构上常出现尺度失真，`_s2_legacy_functions.py` 中的 `refine_dimensions_with_vlm`（配合 `VLM_SIZE_CORRECTION_PROMPT`）利用 VLM 的视觉常识对异常的 OBB 长宽高比例进行提前校准。
2. **兼顾视觉特征与尺寸兼容性的 DINOv2 检索**：系统使用 `dinov2_vitl14.pth` 提取目标裁剪图的图像特征，与同类别候选资产的多视角预计算特征进行比对，并融合尺寸差异惩罚项对候选资产排序（`_process_and_sort_candidates`、`calculate_size_difference`）：
   $$
   score = \text{visual\_similarity} - \alpha \cdot \text{size\_difference}, \quad (\alpha = 0.1).
   $$
   同时，成套分组（Group）内的实例统一复用组内最优资产 ID。
3. **背景材质贴图检索**：`TextureRetrieval` 结合图像块特征与颜色直方图，将墙面、地面和天花板匹配到 `background_texture_dataset` 中的具体材质贴图。这一开源仓库新增模块显著提升了最终 Blender 渲染图的视觉完整度。

## 6. 姿态、位置与尺度估计（S3）：多视角匹配、单应性重排与几何增强

拿到候选 3D 资产后，`PoseModule.run()`（核心逻辑位于 `_s3_legacy_functions.py` 的 `inference_obj_pose()` 及 `utils/view_matching.py`）需估计每个资产在场景中的三维旋转 $R_i$、平移 $t_i$ 与尺度 $s_i$，输出 `{scene_name}_placement_info.json`。

### 6.1 162 视角模板局部特征匹配（AENet）

系统预先为资产库中的每个模型在球面均匀采样的 `162` 个视角下渲染模板图与掩码，并提取形状为 `[N, 162, 1024, 16, 16]` 的姿态敏感特征（`parallel_extract_template_ae_features`）。推理时，借鉴 GigaPose 思路，使用 AENet 与 `LocalSimilarity` 计算待测物体掩码区域 $I_{\mathbf{m}_i}$ 与资产第 $v_k$ 个视角渲染图 $I_{\mathbf{m}_i,v_k}^{A}$ 在有效匹配点集 $\mathcal{K}^{v_k}$ 上的局部余弦相似度之和：

$$
\text{sim}_{img}(I_{\mathbf{m}_i,v_k}^{A}, I_{\mathbf{m}_i})
=
\sum_{j\in\mathcal{K}^{v_k}}
\cos
\left\langle
F_{ae}(I_{\mathbf{m}_i,v_k}^{A})_{img}(j),
F_{ae}(I_{\mathbf{m}_i})_{img}(j)
\right\rangle.
$$

在源码实现中，这一计算由特征批处理、掩码裁剪（mask crop）、模板加载、视角排序（view 排序）与多候选合并协同完成，筛选出粗粒度候选视角集合 $V_{can}$。

### 6.2 单应性矩阵精细重排（Homography Fine Selection）

仅靠局部特征相似度容易在对称家具上产生正反面翻转歧义。论文对粗选视角集 $V_{can}$ 中的匹配点对利用 RANSAC 估计单应性变换（`compute_homography`），对其奇异值分解结果 $U_v, V_v^T$ 计算与单位阵 $I$ 的 Frobenius 范数距离（对应 `extract_rotation`、`homography_rotation_distance`、`cal_homography_rotation_difference`），选出几何形变最小的前 $k$ 个视觉候选视角：

$$
\{v_i^{vis}\}_{i=1}^{k}
=
\arg\min_{v\in V_{can}}^{(k)}
\left\lVert U_vV_v^T-I \right\rVert_F^2.
$$

单应性变换越接近刚体正交旋转，说明模板视角与图像中物体的真实朝向越吻合。

### 6.3 OBB 几何方向增强与小物体回退策略

为进一步提升低角度误差阈值（如 5° 以内）的旋转精度，`find_view_best_match_obb` 将单目点云拟合出的 OBB 方向候选集 $\{v^{obb}\}$ 与视觉候选集 $\{v^{vis}\}$ 进行旋转矩阵夹角比对：

$$
(v_*^{obb},v_*^{vis})
=
\arg\min
\arccos
\left(
\frac{\text{Trace}(R^{v^{vis}T}R^{v^{obb}})-1}{2}
\right),
$$

并根据角度阈值 $\tau=\pi/5$ 决定最终最佳视角：

$$
v_{best}
=
\begin{cases}
v_*^{obb}, & \theta \le \tau, \\
v_1^{vis}, & \theta > \tau.
\end{cases}
$$

当视觉候选与 OBB 几何主轴夹角不超过 $\pi/5$ 时，采纳几何约束更精确的 $v_*^{obb}$；若两者偏差过大，或者目标属于点云稀疏、OBB 朝向不可靠的小物体时，则触发回退策略（fallback），直接采用视觉匹配排名第一的 $v_1^{vis}$。随后通过 `convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates` 将视角姿态转换至 Blender 场景坐标系（如深度轴到 `x, z, -y` 的映射），并结合 OBB 中心、资产原始包围盒与边缘截断标记（mask truncation）初始化平移 $t_i$ 与尺度 $s_i$。

## 7. 布局优化与物理仿真（S4）：局部对齐、体素退火与 Blender 刚体下落

S3 产出的逐物体独立估计不可避免地存在悬空、穿模与墙面夹角偏差。`LayoutModule.run()` 启动独立后台进程执行 `modules/S4_blender_layout_and_corr.py`（内含 `BlenderManager`、`VoxelManager`、`ObjManager`、`RelativePoseManager` 四大管理器），通过四步优化将初始放置转化为物理合理的最终场景。

### 7.1 局部几何修正与容器内部子空间（Internal Placement Spaces）

1. **墙面与朝向局部对齐**：调用 `process_rotation_against_wall_hierarchical`、`process_translation_against_wall`、`process_wall` 与 `process_directly_facing`，强制靠墙物体的背面法线与墙面平齐并贴合墙面，同时修正成对桌椅的相对朝向。
2. **容器内部可放置子空间**：针对书架、敞开式储物柜、托盘等容器类父物体，传统方法仅将子物体简单堆叠在父物体 OBB 顶面。Imaginarium 利用预标注的内部子空间数据，通过 `get_closest_subspace`、`align_obj_to_closest_subspace`、`scale_obj_to_fit_subspace` 与 `resolve_collisions_in_subspace` 将书籍、花瓶与摆件精准嵌入书架隔层或柜体内腔，并自动微调尺度以防穿出隔板。

### 7.2 全局体素碰撞优化（Simulated Annealing）

为同时消除物体穿模、维持支撑接触并尽量保留 2D 引导图中的视觉位置，论文构建了全局平移优化目标：

$$
\min_{\{t_i^{update}\}}
\sum_i
\lambda_1
\left\lVert t_i-t_i^{update}\right\rVert_2^2
+
\left\lVert
\mathbf{m}_i-
\mathcal{R}_{\mathbf{m}}(\text{obj}_{\mathbf{m}_i},v_{ref})
\right\rVert_2^2, \quad (\lambda_1 = 0.1),
$$

并施加四项硬性空间约束（互不穿模、天花板吊挂高度对齐 $t^c$、靠墙距离为零、被支撑物体底面高度贴合支撑物顶面 $z(\text{obj}_{\mathbf{m}_i})^*$）：

$$
\text{obj}_{\mathbf{m}_i}\cap \text{obj}_{\mathbf{m}_j}=\emptyset,
\quad
z(\text{obj}_{\mathbf{m}_i})_{max}=t^c,
\quad
d(\text{obj}_{\mathbf{m}_i},\text{obj}_w)=0,
\quad
z(\text{obj}_{\mathbf{m}_j})_{min}=z(\text{obj}_{\mathbf{m}_i})^*.
$$

由于直接计算高分辨率三角网格的精确交集开销极大，`VoxelManager` 加载预计算的资产体素表示（precomputed voxels）作为几何代理；`ObjManager` 则通过 `calc_overlap_area`、`calc_constraints`、`try_perturb_random_obj` 与模拟退火算法（`simulated_annealing`）迭代搜索无碰撞且满足支撑树的全局最优位置。

### 7.3 Blender 刚体物理下落仿真

在优化的最后阶段，脚本通过 `add_rigid_body` 为场景对象配置刚体碰撞属性，并调用 `scripts.drop_sim_script.run_drop_simulation` 执行重力下落物理仿真。这使得沙发上的抱枕、床上织物以及堆叠的小物件能够自然沉降贴合支撑表面，最终导出优化后的布局文件 `{scene}_placement_info_s4.json` 与渲染图 `{scene}_render_simu.png`。

## 8. 实验评估与消融分析

### 8.1 运行开销与主观/专家评估（Table 1 – Table 2）

**1. 阶段运行耗时分解**

在单张 NVIDIA A100 GPU 上，生成一个完整复杂场景的端到端耗时约为 `240 秒`（对比专业 3D 美术师手工搭建同类场景平均耗时约 `2.5 小时`）：

| 阶段 | 单场景耗时 | 硬件建议与离线预处理成本 |
| --- | ---: | --- |
| Text-to-Image Generation (T2I) | `10s` | 推荐 A100；I2Layout 阶段可在 RTX 3090 及以上运行 |
| Scene Image Analysis (S0 + S1) | `110s` | 若自行从头生成派生数据：多视角模板渲染约需 `1–2 天` |
| Scene Layout Reconstruction (S2 + S3) | `60s` | AENet 特征本地提取约需 `2 小时`，FBX 转 Blend 约需 `20 分钟` |
| Layout Refinement (S4) | `60s` | 直接下载官方 Hugging Face 派生数据包可跳过上述离线预处理 |

**2. 艺术专业学生偏好研究（Table 1）与专家/GPT-4o 评分（Table 2）**

论文邀请 `100` 名高年级艺术专业学生，在餐厅、客厅、卧室三类场景上对 Imaginarium 与四大基线进行双盲成对比较（评估合理真实性 Q1 Reasonable/Realistic 与审美连贯性 Q2 Coherent/Aesthetic），表内数值为相比基线选择 Imaginarium 的偏好百分比：

| 对比基线 | Reasonable Dining | Reasonable Living | Reasonable Bedroom | Reasonable 平均 | Aesthetic Dining | Aesthetic Living | Aesthetic Bedroom | Aesthetic 平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffuScene | 75.69% | 82.59% | 79.37% | **79.22%** | 74.86% | 85.57% | 80.72% | **80.38%** |
| HOLODECK | 79.27% | 77.08% | 76.79% | **77.71%** | 82.72% | 72.92% | 74.55% | **76.73%** |
| LayoutGPT | N/A | 76.69% | 76.50% | **76.60%** | N/A | 77.54% | 81.11% | **79.33%** |
| InstructScene | 66.33% | 68.46% | 61.29% | **65.36%** | 69.39% | 75.17% | 72.90% | **72.49%** |

同时，论文邀请 `20` 位具有三年以上经验的专业 3D 艺术家与 GPT-4o 对 `60` 个生成场景从构图（Composition）、语义逻辑（Semantic）与审美吸引力（Aesthetic）三个维度打分（5 分制，格式为 `专家评分 / GPT-4o 评分`）：

| Method | Composition | Semantic | Aesthetic | Overall |
| --- | --- | --- | --- | --- |
| **Imaginarium (Ours)** | **3.35 / 3.16** | **3.29 / 2.86** | **3.37 / 3.16** | **3.34 / 3.06** |
| InstructScene | 2.91 / 3.07 | 2.75 / 2.83 | 2.89 / 3.08 | 2.85 / 2.99 |
| DiffuScene | 2.86 / 3.07 | 2.80 / 2.78 | 2.83 / 3.07 | 2.83 / 2.97 |
| HOLODECK | 2.71 / 2.91 | 2.56 / 2.55 | 2.80 / 2.86 | 2.69 / 2.77 |
| LayoutGPT | 2.42 / 2.97 | 2.26 / 2.83 | 2.35 / 2.97 | 2.34 / 2.92 |

专业人类评审在各维度上拉开了比 GPT-4o 自动评分更显著的差距，印证了 2D 视觉引导结合高质量资产库在构图层次与陈设丰富度上的优势。

### 8.2 重建保真度与 3DF-CLAPE 旋转估计基准（Table 3 – Table 4）

在随机抽取的 `30` 个数据集真值场景渲染图上，Table 3 分别统计了结构关键的主要物体（Primary，如地面/天花板支撑及靠墙家具）与次要装饰小物体（Secondary）的恢复保真度：

| 评估指标 | Primary（主要物体） | Secondary（次要物体） | 全局综合指标 |
| --- | ---: | ---: | --- |
| Object Recovery（物体恢复率） | 92.31% | 70.41% | Scene Graph Accuracy：`93.26%` |
| Category Preservation（类别保持率） | 95.83% | 91.67% | CLIP Score（Guide / Render）：`27.03` / `25.83` |
| Rotation AUC@60°（旋转精度） | 74.83% | 71.51% | GPT-4o 综合评分：`8.29 / 10` |
| Translation AUC@0.5m（平移精度） | 84.32% | 80.40% | — |

针对单图三维旋转估计，作者构建了包含类别级（Category-level）与实例级（Instance-level）两个子集的 **3DF-CLAPE** 基准（Table 4，指标为 AUC@60°）：

| Method | Category-level AUC@60° ↑ | Instance-level AUC@60° ↑ |
| --- | ---: | ---: |
| DiffCAD | 26.45% | 25.44% |
| DINOv2 | 31.68% | 31.38% |
| GigaPose | 39.85% | 57.43% |
| AENet | 45.32% | 62.16% |
| SPARC | 52.54% | 61.46% |
| Orient Anything | 56.07% | 56.24% |
| **Imaginarium (Ours)** | **70.06%** | **81.44%** |

融合 162 视角 AENet 匹配、单应性重排与 OBB 几何方向增强后，旋转估计精度相比单独使用 GigaPose 或 AENet 提升了近 20 个百分点。

### 8.3 核心模块消融实验（Table 5 – Table 8）

**1. Flux 微调对资产检索精度与多样性的影响（Table 5 & Table 6）**

在 `100` 张生成场景图的人工标注检索评测中，微调 Flux 显著提升了下游资产匹配的命中率，同时保持了良好的生成多样性（并未退化为死记硬背训练集布局）：

| 模型变体 | 检索 Top-1 ↑ | 检索 Top-3 ↑ | NN LPIPS ↑ | 训练集场景相似度 ↓ | DIV LPIPS ↑ | 集合内场景相似度 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla Flux（未微调） | 48.57% | 68.57% | 0.6375 | 0.3665 | 0.5782 | 0.2974 |
| **Finetuned Flux（LoRA 微调）** | **68.70%** | **83.21%** | 0.5981 | 0.3899 | **0.5901** | 0.3178 |

**2. S3 旋转估计子模块消融（Table 7）**

| AENet 粗匹配 | 单应性重排（Homography） | OBB 几何增强（Geometry） | mAP@5° ↑ | mAP@15° ↑ | mAP@45° ↑ |
| :---: | :---: | :---: | ---: | ---: | ---: |
| ✓ | ✗ | ✗ | 4.30% | 15.34% | 67.92% |
| ✓ | ✓ | ✗ | 5.21% | 59.42% | 76.07% |
| ✓ | ✗ | ✓ | 36.22% | 71.73% | 77.16% |
| **✓** | **✓** | **✓** | **66.57%** | **75.28%** | **80.61%** |

仅用 AENet 时 `mAP@5°` 仅为 `4.30%`；引入单应性重排使 `mAP@15°` 从 `15.34%` 跃升至 `59.42%`；叠加 OBB 几何方向增强后，高精度阈值 `mAP@5°` 大幅提升至 **66.57%**，证实了视觉局部特征与三维点云几何的强互补性。

**3. S4 布局优化阶段递进消融（Table 8）**

| 优化阶段 | 支撑正确率（Support Correctness）↑ | 平均穿模对数（Intersection Pairs）↓ | GPT-4o 评分 ↑ |
| --- | ---: | ---: | ---: |
| Initial Estimation（S3 初始输出） | 62.45% | 5.43 | 2.83 |
| + Local Refinement（局部墙面/朝向修正） | 72.86% | 4.43 | 3.07 |
| + Global Optimization（体素模拟退火） | 90.80% | 2.21 | 3.26 |
| **+ Physical Constraints（Blender 物理下落）** | **91.34%** | **2.20** | **3.29** |

全局模拟退火优化贡献了最大的物理可用性跃升（支撑正确率提升近 18 个百分点，穿模对数减半），随后的刚体物理仿真则进一步消除了细微悬空。

## 9. 交互编辑、横向对比与系统局限

### 9.1 基于 2D 局部重绘的 3D 场景交互编辑

由于 2D 引导图是系统的核心控制面，Imaginarium 天然支持“渲染回 2D $\to$ 图像局部重绘 $\to$ 增量更新 3D 布局”的交互编辑工作流：

```text
已有 3D scene
  -> render global/local image
  -> mask target region
  -> Flux repaint/inpaint (or Canny control)
  -> parse edited region
  -> re-retrieve assets & re-estimate pose
  -> merge back into fixed scene
```

用户可通过掩码指定局部区域并结合文本提示完成全局场景补全（global scene completion）、单物体替换（object replacement）或局部新增摆件（local object addition），未编辑区域的 3D 资产则保持锁定。

### 9.2 与 CasLayout 及相关工作的技术谱系对比

将 Imaginarium 与同领域的代表性室内场景生成方法置于同一维度对比：

| 方法 | 主要生成范式 | 使用 2D 引导图 | 资产库依赖 | 显式场景图 | 核心局限 |
| --- | --- | :---: | :---: | :---: | --- |
| [ATISS](https://arxiv.org/abs/2110.03675) | 自回归室内布局生成 | 否 | 强依赖 | 否 | 受限于训练集房型，缺乏细粒度摆件与视觉引导 |
| [DiffuScene](https://arxiv.org/abs/2303.14207) | 单阶段扩散布局生成 | 否 | 强依赖 | 弱 | 联合分布学习难度高，陈设密度与资产类别有限 |
| [InstructScene](https://openreview.net/forum?id=ELrEzO9DEq) | 文本指令 + 语义图先验扩散 | 否 | 强依赖 | 是 | 文本到复杂几何与小物体层级的映射受限 |
| [LayoutGPT](https://arxiv.org/abs/2305.15393) | LLM 直接规划坐标 | 否 | 强依赖 | 隐式 | 连续几何精度不足，易发生穿模或越界 |
| [HOLODECK](https://arxiv.org/abs/2312.09067) | LLM + 规则约束求解 | 否 | 强依赖 | 规则约束 | 审美构图与复杂多层支撑物的匹配能力有限 |
| SceneCraft | LLM 生成 Blender 脚本 | 否 | 视配置而定 | 隐式 | 脚本执行稳定性与复杂几何控制受限 |
| LayoutVLM | VLM + 可微优化 | 部分支持 | 强依赖 | 部分支持 | 强依赖初始布局与可微渲染优化收敛性 |
| CAST | 单图对象生成 + 姿态对齐 | 是 | 生成式资产 | 否 | 生成网格的拓扑规范性与工业资产复用性不足 |
| [CasLayout](https://arxiv.org/abs/2604.27361) | 四阶段级联扩散 + Relation VAE | 否（可选输入） | CAD 检索 | 稀疏功能图 | 依赖 3D-FRONT 训练分布，小物件层级覆盖较少 |
| **Imaginarium** | **2D 视觉引导 + 检索姿态 + 物理优化** | **是** | **强依赖（高质量库）** | **是（支撑/靠墙/成组）** | **单图遮挡歧义、有限资产库拓扑边界、管线较重** |

对比可见，**CasLayout** 代表了“在 3D 布局空间内通过级联扩散与隐式关系潜变量进行生成建模”的路线，强项在于户型门窗约束与毫秒级采样；而 **Imaginarium** 则代表了“利用 2D 扩散模型生产高审美构图，再通过视觉解析、6D/9D 姿态匹配与 Blender 物理引擎落地为高密度 3D 资产”的图形系统工程路线，单场景平均物体数（`31.86`）与视觉丰富度显著面向工业内容生产。

### 9.3 系统局限与失败边界

1. **有限资产库带来的新拓扑失配（Novel Topology Mismatch）**：如论文附录失败案例所示，当 Flux 生成了资产库未收录的复合结构（如开放书架与封闭衣柜的结合体）时，S2 只能检索普通封闭衣柜，导致原本应放置在开放隔层中的子物体失去合法内部子空间。
2. **单视角严重遮挡下的姿态不适定性**：当物体仅露出局部边缘或无纹理背面时，162 视角匹配与单应性重排仍存在多解歧义；引入多视角一致生成是未来的自然延伸方向。
3. **长链路误差传播与计算/环境成本**：从 `Flux -> VLM -> Grounding-DINO -> SAM -> Depth Anything V2 -> RANSAC -> DINOv2 -> AENet -> Blender` 的超长管线意味着上游的漏检或平面拟合翻转会直接传递至下游；同时，单场景约 4 分钟的耗时及多套外部 API、预计算缓存与 Blender 依赖也构成了较高的部署门槛。

## 10. 工程复现指南与总结

### 10.1 环境准备与两阶段运行命令

完整复现或部署开源管线需准备 Python `3.10`、Blender `4.3.2`（`4.0+` 兼容）、LLM/VLM API 密钥与 Grounding-DINO Token，从 `config/config-example.yaml` 复制生成 `config/config.yaml`，并按第 3.2 节目录结构解压基础资产包、派生数据包与四个核心预训练权重（Flux LoRA、DINOv2、AENet、Depth Anything V2）。

推理分两步执行：

```bash
# Stage 1: Text-to-Image 生成 2D 引导图
python run_imaginarium_T2I.py \
  --prompt 'A cozy living room featuring comfortable armchairs, a gallery wall, and a stylish coffee table.' \
  --num 4

# Stage 2: Image-to-3D Layout 解析与重建（支持默认断点恢复、--clean 重跑与 --debug 导出诊断图）
python run_imaginarium_I2Layout.py demo/demo_0.png --clean --debug
```

### 10.2 G0–G7 阶段式质量门禁与逆向排错法

面对多模块串联系统，建议按以下 8 个检查门禁（Gates）逐步核验中间产物，一旦最终渲染出现异常即可按因果链逆向定位：

| Gate | 检查对象 | 通过标准 | 异常表现与优先排查位置 |
| --- | --- | --- | --- |
| **G0** | 输入引导图 | 1024 分辨率、透视稳定、核心家具清晰可见 | 若构图杂乱或视角极端，重新采样 T2I 或更换引导图 |
| **G1** | `depth.png` (S0) | 地面、墙面与前景家具深度层次合理 | 若整个房间尺度畸变，检查 Depth 权重与 `estimate_focal_length` 焦距假设 |
| **G2** | `masks.pkl` (S1) | 核心家具无漏检，掩码未大面积吞并背景 | **最终场景缺失某物体**时优先查此处：检查 VLM 物体列表、Grounding-DINO 阈值与 SAM 掩码 |
| **G3** | `floor_walls_pose.json` (S1) | 地面水平、墙面法线与图像透视一致 | **房间整体倾斜或靠墙方向全错**时优先查此处：检查 RANSAC 点云输入与背景掩码过滤 |
| **G4** | `scene_graph_result_final.json` (S1) | `parent`、`isAgainstWall`、`directlyFacing` 与 `group` 合理 | 若椅子未围桌或子物体未归属桌面，检查 `used_prompts.py` 输出与二次核查逻辑 |
| **G5** | `retrieval_results_final.json` (S2) | Top-k 候选的类别、风格与尺寸与裁剪图契合 | **物体类别对但外观/尺寸怪异**时优先查此处：检查 DINOv2 特征、资产分类粒度与 VLM 尺寸修正 |
| **G6** | `pose_prediction_stitched.png` (S3) | 渲染模板视角与裁剪图对齐，无 180° 翻转 | **物体外观对但朝向反向**时优先查此处：检查单应性重排、OBB 角度阈值 $\tau$ 与小物体回退逻辑 |
| **G7** | `{scene}_render_simu.png` (S4) | 无明显穿模与悬空，墙地贴图与内部空间放置自然 | **位置大体正确但局部穿模/悬空**时查此处：检查预计算体素、内部子空间与刚体下落参数 |

### 10.3 总结

Imaginarium 展示了一条极具工程实用价值的 3D 场景生成路径：

```text
高质量 asset library
+ fine-tuned Flux guide image
+ VLM semantic parsing & scene graph
+ Grounding-DINO / SAM masks
+ Depth Anything V2 / RANSAC geometry
+ DINOv2 / AENet retrieval and 162-view pose matching
+ voxel simulated annealing & Blender rigid-body physics
```

在 3D 场景真值数据稀缺而 2D 视觉生成模型高度成熟的当下，将微调后的 2D 引导图像作为中间视觉控制面，再通过视觉基础模型解析、多视角几何姿态匹配与图形学物理优化落地为高质量 3D 资产布局，为兼顾视觉审美丰富度与工业级资产可控复用提供了清晰的范本。

## 参考文献

- Zhu et al., 2025. [Imaginarium: Vision-guided High-Quality 3D Scene Layout Generation](https://arxiv.org/abs/2510.15564).
- Hugging Face Paper Page. [Imaginarium metadata](https://huggingface.co/papers/2510.15564).
- HiHiAllen. [Imaginarium official repository](https://github.com/HiHiAllen/Imaginarium).
- Imaginarium project page. [Vision-guided High-Quality 3D Scene Layout Generation](https://ydove0324.github.io/Imaginarium/).
- Hugging Face Dataset. [HiHiAllen/Imaginarium-Dataset](https://huggingface.co/datasets/HiHiAllen/Imaginarium-Dataset).
- Hugging Face Dataset. [binicey/Imaginarium-3D-Derived-Dataset](https://huggingface.co/datasets/binicey/Imaginarium-3D-Derived-Dataset).
- Paschalidou et al., 2021. [ATISS: Autoregressive Transformers for Indoor Scene Synthesis](https://arxiv.org/abs/2110.03675).
- Tang et al., 2024. [DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis](https://arxiv.org/abs/2303.14207).
- Lin and Mu, 2024. [InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior](https://openreview.net/forum?id=ELrEzO9DEq).
- Feng et al., 2024. [LayoutGPT: Compositional Visual Planning and Generation with Large Language Models](https://arxiv.org/abs/2305.15393).
- Yang et al., 2024. [HOLODECK: Language Guided Generation of 3D Embodied AI Environments](https://arxiv.org/abs/2312.09067).
- Dhamo et al., 2021. [Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs](https://openaccess.thecvf.com/content_ICCV_2021/html/Dhamo_Graph-to-3D_End-to-End_Generation_and_Manipulation_of_3D_Scenes_Using_Scene_ICCV_2021_paper.html).
- Zhai et al., 2024. [CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs](https://arxiv.org/abs/2305.16283).
- Nguyen et al., 2024. [GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence](https://arxiv.org/abs/2311.14155).
- Yang et al., 2024. [Depth Anything V2](https://arxiv.org/abs/2406.09414).
- Kirillov et al., 2023. [Segment Anything](https://arxiv.org/abs/2304.02643).
- Ren et al., 2024. [Grounding DINO 1.5 API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API).
- Ruiz et al., 2023. [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242).
- Black Forest Labs. [FLUX.1](https://github.com/black-forest-labs/flux).
