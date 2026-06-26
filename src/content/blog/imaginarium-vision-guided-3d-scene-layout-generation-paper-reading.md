---
title: "Imaginarium 论文精读：视觉引导、高质量资产库、代码实现与 3D 场景布局生成"
description: "从论文与官方代码双线精读 Imaginarium 如何通过 Flux 图像引导、视觉解析、资产检索、姿态估计、场景图约束和 Blender 物理优化生成高质量 3D 场景布局"
pubDate: "2026-06-26T10:11:20+08:00"
updatedDate: "2026-06-26T10:11:20+08:00"
tags:
  - "Paper Reading"
  - "3D Scene Synthesis"
  - "Computer Vision"
  - "Generative AI"
  - "Graphics"
  - "Code Reading"
draft: false
---

这篇报告精读的论文是 [Imaginarium: Vision-guided High-Quality 3D Scene Layout Generation](https://arxiv.org/abs/2510.15564)。论文由 Xiaoming Zhu、Xu Huang、Qinghongbing Xie、Zhi Deng、Junsheng Yu、Yirui Guan、Zhongyuan Liu、Lin Zhu、Qijun Zhao、Ligang Liu、Long Zeng 撰写，arXiv 提交日期是 2025 年 10 月 17 日，arXiv DOI 是 `10.48550/arXiv.2510.15564`。官方仓库 README 标注其为 **SIGGRAPH Asia 2025 & ACM Transactions on Graphics (TOG)** 工作，并提供了 [项目页](https://ydove0324.github.io/Imaginarium/)、[官方代码仓库](https://github.com/HiHiAllen/Imaginarium)、[Hugging Face 数据集](https://huggingface.co/datasets/HiHiAllen/Imaginarium-Dataset) 与 [派生数据集](https://huggingface.co/datasets/binicey/Imaginarium-3D-Derived-Dataset)。

一句话概括，Imaginarium 不是一个“从文本直接生成 3D mesh”的模型，而是一个把 **2D 图像生成模型的审美、构图和组合能力** 转译成 **可复用 3D 资产布局** 的系统：

```text
文本提示
  -> 微调 Flux 生成 2D guide image
  -> VLM / 检测 / 分割 / 深度估计解析图像
  -> 从高质量资产库检索 3D asset
  -> 估计旋转、平移、缩放
  -> 用 scene graph、优化和 Blender physics 修正布局
  -> 输出可渲染的 3D scene layout
```

本文会同时读论文和官方代码。也就是说，讨论每个模块时不只问“论文怎么说”，还会问：

1. 代码入口在哪里？
2. 输入输出文件是什么？
3. 论文中的概念在代码里变成了哪些阶段、函数和配置项？
4. 当前代码和论文原始叙述有哪些差异？
5. 如果要复现或工程化，真正卡人的地方在哪里？

需要先说明复现状态。官方仓库已经不是 “coming soon”：README 给出了安装、数据、权重、Blender、T2I、I2Layout 的说明。但本文不下载大数据集、不运行完整 pipeline、不声称完成复现。本文做的是论文级精读和源码级静态阅读；代码依据是官方 GitHub `main` 分支当前公开源码。

## 1. 读者导读：Imaginarium 到底解决什么

3D scene layout generation 的目标不是生成一张静态图片，而是生成一个由资产、位置、朝向、尺度和关系组成的 3D 场景。它要服务的是游戏场景、CGI、虚拟拍摄、室内外设计、仿真数据生成等任务。对这类任务来说，输出必须能被渲染、编辑、复用和检查。

Imaginarium 的目标可以拆成四层。

| 层级 | 问题 | Imaginarium 的做法 |
| --- | --- | --- |
| 内容层 | 这个场景应该有什么物体 | 用文本提示和 2D 图像生成扩展场景内容 |
| 资产层 | 用哪个 3D 模型表示物体 | 从预定义高质量 asset library 检索 |
| 几何层 | 资产如何摆放 | 从图像分割、深度、OBB、姿态匹配估计变换 |
| 逻辑层 | 布局是否合理 | 用 scene graph、support tree、wall constraints、physics 修正 |

这和很多 text-to-3D 或 scene synthesis 工作的差别很大。它不是让一个 LLM 直接吐出坐标，也不是让扩散模型端到端生成每个 object token，而是把 2D 图像作为中间表示：先让图像模型生成一个视觉上丰富、构图合理、风格一致的 reference image，再把这个 reference image 反解析成 3D 布局。

这个设计很务实。2D 图像模型拥有海量图像先验，擅长构图、风格、细节密度和视觉层次。3D scene layout 数据很少，而且资产库通常有限。如果直接训练 3D layout generator，数据规模和资产质量会成为瓶颈。Imaginarium 的选择是：让 2D 模型负责想象画面，让视觉解析和几何优化负责把画面落回 3D asset world。

## 2. 论文信息与代码状态

论文的正式题名是 **Imaginarium: Vision-guided High-Quality 3D Scene Layout Generation**，HF paper metadata 和 arXiv abstract 均一致。论文摘要给出的主线是：

- 构建一个高质量 asset library，包含 `2,037` 个 scene assets 和 `147` 个 3D scene layouts。
- 使用图像生成模型扩展 prompt 到 image，并微调它以对齐资产库。
- 设计 image parsing module，从视觉语义和几何信息恢复 3D layout。
- 使用 scene graph 和整体视觉语义优化布局，使其逻辑一致并贴近图像。
- 通过用户测试证明布局丰富度和质量优于现有方法。

官方 GitHub 仓库当前公开了完整工程骨架，包括：

```text
run_imaginarium_T2I.py
run_imaginarium_I2Layout.py
pipeline.py
config/config-example.yaml
core/context.py
modules/geometry.py
modules/parsing.py
modules/retrieval.py
modules/pose.py
modules/layout.py
modules/S4_blender_layout_and_corr.py
modules/_s1_legacy_functions.py
modules/_s2_legacy_functions.py
modules/_s3_legacy_functions.py
utils/obb.py
utils/ransac.py
utils/view_matching.py
prompts/used_prompts.py
```

README 里把使用流程分成两个用户可见阶段：

```text
Stage 1: Text-to-Image (T2I)
  python run_imaginarium_T2I.py --prompt ... --num 4

Stage 2: Image-to-3D Layout (I2Layout)
  python run_imaginarium_I2Layout.py demo/demo_0.png --clean --debug
```

但代码内部的 I2Layout pipeline 实际分成五个阶段：

```text
S0_geometry
S1_parsing
S2_retrieval
S3_pose
S4_layout
```

这是本文的第一个重要观察：论文里的方法结构是面向叙事和研究贡献组织的，代码里的阶段结构是面向工程执行和中间结果缓存组织的。读这篇论文如果只看 Fig. 2，会漏掉很多系统工程细节；读代码如果只看 `pipeline.py`，又会低估论文中“2D 图像先验如何迁移到 3D”的设计动机。

## 3. 背景与任务边界

为了读懂 Imaginarium，需要先区分几个相近但不等价的概念。

**3D scene layout** 指的是场景中对象的类别、资产实例、旋转、平移、缩放、层级关系、支撑关系和空间约束。它通常不从零生成每个 mesh，而是摆放已有资产。

**Predefined asset library** 是一组可用 3D 模型。每个资产可能有 mesh、材质、尺寸、类别、渲染视角、embedding、内部可放置空间等元数据。Imaginarium 的输出必须从这个资产库里选择对象，因此它天然适合游戏和 CGI 的工业资产复用。

**Asset retrieval** 是把图像中的一个 segmented object 匹配到资产库中的 3D 模型。它不是简单按类别选模型，还要考虑视觉相似度、尺寸兼容和风格一致性。

**Pose estimation** 在这里不是经典 6D pose benchmark 的封闭对象姿态，而是给定一张生成图像和一个候选 CAD asset，估计这个 asset 应该在 3D 场景里的旋转、位置和尺度。

**OBB** 是 oriented bounding box，带朝向的包围盒。Imaginarium 用单图深度和 mask 得到前景物体点云，再拟合 OBB，作为 translation、scale、scene graph refinement 和 layout optimization 的几何锚点。

**Scene graph** 表示对象之间的支撑、靠墙、吊挂、朝向、分组等关系。Imaginarium 的 scene graph 不是完整的语义知识图谱，而是为了让布局合理而提取的一组可执行空间约束。

**Physical plausibility** 指对象不能无支撑悬空、不能严重穿模、需要与墙/地板/天花板接触合理，柔性或堆叠对象需要经过物理仿真修正。

因此，Imaginarium 的边界很清楚：它生成的是 **基于有限资产库的 3D scene layout**，不是开放世界的 3D mesh 生成，也不是单纯文本规划。这个边界既是优势，也是限制。优势是资产可控、可复用、可渲染；限制是图像里出现资产库没有的 novel topology 时，检索和后续布局都会出错。

## 4. 问题形式化：从 prompt 到对象变换

论文把任务写成一个从文本提示和资产库到场景对象集合的函数：

$$
G(O \mid prompt, A)=\{o_1,o_2,o_3,\dots,o_n\},
$$

其中 $A$ 是预定义资产库，$prompt$ 是用户输入文本。每个对象 $o_i$ 可以写成：

$$
o_i=\{\text{obj}_i,R_i,t_i,s_i\}.
$$

这里：

| 符号 | 含义 | 代码侧对应 |
| --- | --- | --- |
| $\text{obj}_i$ | 从资产库选中的 3D asset | `retrieval_results_final.json` 中的 asset id/name |
| $R_i$ | asset 旋转矩阵或姿态 | `S3_pose_inference` 生成的 pose matrix |
| $t_i$ | 3D translation | `placement_info.json` 中的位置字段 |
| $s_i$ | scale | S3/S4 根据 OBB、资产尺寸、容器空间修正 |
| $A$ | asset library | `asset_data/imaginarium_assets/` 与 `imaginarium_asset_info.csv` |
| $O$ | 输出 object set | S4 后的 refined placement info 与 Blender scene |

这个公式看起来简单，但它隐藏了两个难点。

第一，$prompt$ 到 $O$ 的映射不是一对一。一个 “cozy living room” 可以有无数种合理布局。Imaginarium 不直接让语言模型决定坐标，而是先生成 2D guide image，把语言空间压到一个具体视觉方案。

第二，$\text{obj}_i$ 必须来自资产库 $A$。如果 2D guide image 生成了一个“书柜和衣柜混合体”，而资产库只有普通衣柜，那么检索会选最接近的资产，但 scene graph 中依赖开放书架的子物体就可能无法放置。论文附录的 failure case 就指出了这类 semantic-structural mismatch。

**论文-代码对照。** 代码没有直接暴露一个名为 `G` 的函数，而是把它拆成 T2I 和 I2Layout 两个 CLI：

```text
run_imaginarium_T2I.py
  prompt -> generated image

run_imaginarium_I2Layout.py
  image -> placement info -> Blender layout
```

I2Layout 中真正串联阶段的是 `ImaginariumPipeline.run()`：

```text
GeometryModule.run()
SemanticParsingModule.run()
RetrievalModule.run()
PoseModule.run()
LayoutModule.run()
```

所以，如果用代码语言重写论文公式，更接近：

$$
G(O \mid prompt,A)
=
\text{I2Layout}(\text{T2I}(prompt;A),A).
$$

这里的 `T2I(prompt; A)` 不是数学上显式把资产库输入 Flux，而是通过对资产库渲染数据进行 LoRA 微调，让图像生成分布更接近资产库风格和构图习惯。

## 5. 总体 pipeline：论文图 vs 代码执行图

论文 Fig. 2 给出的 pipeline 是：

```text
Text prompt
  -> fine-tuned image generation model
  -> 2D guide image
  -> semantic / geometric / relational image analysis
  -> asset retrieval
  -> transformation estimation
  -> scene graph + physical refinement
  -> final 3D layout
```

代码实际执行结构如下：

| 论文模块 | 代码入口 | 主要输出 | 作用 |
| --- | --- | --- | --- |
| Prompt Expander | `run_imaginarium_T2I.py` | `generated_images/*.png` | 文本到 2D guide image |
| Geometry Analysis | `modules/geometry.py` | `S0_geometry_pred_results/depth.png` | Depth Anything V2、intrinsics、depth |
| Semantic Parsing | `modules/parsing.py` | `scene_graph_result.json`, `masks.pkl`, `floor_walls_pose.json` | VLM、Grounding-DINO、SAM、RANSAC、scene graph |
| Asset Retrieval | `modules/retrieval.py` | `pcd_obb_data.json`, `retrieval_results_final.json` | OBB、DINOv2 检索、尺寸修正、贴图检索 |
| Pose Estimation | `modules/pose.py` | `{scene}_placement_info.json` | 162 视角匹配、homography、OBB enhancement |
| Layout Refinement | `modules/layout.py` + Blender script | `{scene}_placement_info_s4.json`, render | wall/support/collision/physics 修正 |

这张表能解释一个容易误读的点：论文里的 “scene image analysis” 在代码里跨了 S0 和 S1；论文里的 “scene layout reconstruction” 在代码里跨了 S2 和 S3；论文里的 “refinement” 在代码里基本集中到 S4，而且 S4 是通过独立 Blender 子进程运行的。

**论文-代码对照。** `pipeline.py` 有一个很工程化的设计：模块导入是 lazy import，注释里明确说移除了 top-level imports 以加快初始加载。`Context` 负责共享模型、日志、输出目录和中间数据。每个 stage 还会检查已有输出，如果不是 `--clean` 模式，就跳过已经完成的阶段。这说明 Imaginarium 的官方代码考虑了重型 pipeline 的调试成本：一次跑完整流程可能很慢，因此 resume、debug artifact、stage log 都是必要设计。

### 5.1 中间文件契约：代码真正如何“传话”

读 Imaginarium 的源码时，一个比模型结构更重要的问题是：每个 stage 到底把什么东西交给下一个 stage。论文图把流程画成连续箭头，但代码里真正连接模块的是一组文件契约和 `Context` 内存字段。如果这些契约读不清，后面很容易把 “semantic parsing 错了” 误判为 “pose estimation 不准”，或者把 “asset retrieval 不匹配” 误判为 “Blender 优化失败”。

下面这张表可以当作源码阅读时的导航图：

| 阶段 | 输入 | 关键中间文件或 Context 字段 | 下游如何使用 | 常见失败信号 |
| --- | --- | --- | --- | --- |
| `run_imaginarium_T2I.py` | text prompt、Flux base model、LoRA | generated guide image | 作为 I2Layout 的视觉目标 | 图像对象太多、视角不稳定、资产库风格不一致 |
| `S0_geometry` | guide image、Depth Anything V2 权重 | `depth.png`、`depth_image`、`intrinsics`、debug `pcd.ply` | S1 的 floor/wall RANSAC、S2 的 OBB 估计、S3 的深度对齐 | 深度尺度漂移、墙地交界不清、焦距近似导致点云拉伸 |
| `S1_parsing` | image、depth、VLM、Grounding-DINO、SAM | `masks.pkl`、`final_detect_items.pkl`、`floor_walls_pose.json`、`scene_graph_result*.json` | S2 根据 mask crop 检索资产；S3/S4 根据 scene graph 约束摆放 | 漏检、重复检测、support parent 错、wall relation 错 |
| `S2_retrieval` | masks、scene graph、asset metadata、DINOv2 features | `pcd_obb_data.json`、`retrieval_results_final.json`、background texture result | S3 用候选资产和 OBB 做姿态估计；S4 读取 mesh/texture | 类别对但风格错、尺寸偏差、同组物体检索不一致 |
| `S3_pose` | retrieval result、OBB、depth、template views、AENet | `{scene}_placement_info.json`、debug pose comparison images | S4 的初始对象变换、支撑关系和遮挡处理 | 旋转翻转、截断物体漂移、小物体 fallback 不稳定 |
| `S4_layout` | placement info、voxels、internal placement space、Blender | `{scene}_placement_info_s4.json`、`{scene}_render_simu.png` | 最终布局和渲染结果 | 穿模、悬空、墙面对齐错误、物理掉落后偏离图像 |

这个文件契约有两个特点。

第一，stage 之间并不是完全端到端可微的。T2I 是生成模型，S0/S1/S2/S3/S4 是视觉模型、VLM、检索、几何启发式和 Blender 物理的组合。它更像一个可恢复、可调试的 production pipeline，而不是一个单一神经网络。

第二，许多中间文件同时承担“程序输入”和“人工审查证据”的角色。例如 `pose_prediction_stitched.png` 不只是 debug 图片，也是判断 S3 是否值得进入 S4 的依据；`scene_graph_result_final.json` 不只是 S4 的输入，也是判断 VLM relation 是否过度幻觉的证据；`pcd_obb_data.json` 不只是几何参数，也是检查 depth/focal length 估计是否把对象尺度拉坏的入口。

**论文-代码对照。** 论文强调 vision-guided layout generation，但源码显示这个 “vision-guided” 不是只在开头看一次图像。图像信息至少被使用了四次：T2I 生成 guide image，S1 用图像做 object parsing，S2 用 crop 做 asset retrieval 和 texture retrieval，S3 用 crop/template matching 估计 pose。换句话说，视觉引导贯穿 pipeline，而不是一个单独的 conditioning token。

### 5.2 运行模式：resume、clean、debug 的工程含义

README 里给出 `--clean` 和 `--debug`，源码里这两个 flag 不是普通开关，而是重 pipeline 可用性的基础。

`resume` 是默认模式。`Context` 会把输出目录固定为 `saved_results/{image_name}_result/`，每个 stage 看到自己需要的结果文件已经存在时就跳过。这对 Imaginarium 特别关键，因为一次运行里包含 VLM/API 调用、DINOv2 特征、AENet matching、Blender 子进程和物理仿真。任何一个中间环节失败，如果不能 resume，调试成本会非常高。

`--clean` 是强制从头开始。它适合三种情况：输入 guide image 变了；配置中的权重、API 或资产路径变了；上一次中间文件明显被污染，例如旧的 mask 对应旧图像、旧的 retrieval result 对应新资产库。

`--debug` 会保存更多可视化和匹配中间结果。它会变慢，但对论文精读很有价值，因为 Imaginarium 的许多错误不是最终 render 才能看出来，而是在 S1/S2/S3 已经能定位。例如：

- 如果 `masks.pkl` 漏掉主要家具，后面的检索和姿态估计没有修复空间。
- 如果 floor/wall RANSAC 错，把墙当地或把地当墙，S4 的 wall alignment 会朝错误方向修。
- 如果 retrieval top candidate 长相不对，S3 pose matching 即使数学上得分高，最终场景也会风格不一致。
- 如果 pose comparison image 显示模板视角与 crop 差异很大，S4 的物理优化只能修碰撞，不能恢复语义视角。

所以，读代码时可以把 `--debug` 理解为论文实验图之外的“可解释性接口”。它不改变论文主方法，但它决定了一个研究者能否真正定位失败原因。

## 6. T2I：Flux prompt expander 代码精读

论文的第一个关键主张是：先把文本 prompt 扩展成 2D guide image。这个设计不是为了生成最终图片，而是为了给后续 3D layout reconstruction 提供一个视觉目标。

论文选择微调 Flux，原因有三点：

1. Flux 这类图像生成模型拥有丰富的构图和视觉细节先验。
2. 微调后可以让生成图像更贴近自有 asset library 的风格、材质和视角。
3. 与直接从 3D 数据训练 layout generator 相比，2D 图像模型能利用更强的视觉生成能力。

官方代码里对应的是 `run_imaginarium_T2I.py`。核心参数包括：

| 参数 | 默认值或含义 | 解释 |
| --- | --- | --- |
| `--base_model` | `black-forest-labs/FLUX.1-dev` | 基础 Flux 模型 |
| `--lora_path` | `weights/imaginarium_finetuned_flux.pth` | 资产库风格微调权重 |
| `--guidance_scale` | `7.5` | 文本引导强度 |
| `--num_inference_steps` | `50` | 采样步数 |
| `--resolution` | `1024` | 输出图像分辨率 |
| `--num` | 生成张数 | 可一次生成多个候选 guide image |

代码中的关键伪流程是：

```python
transformer = FluxTransformer2DModel.from_pretrained(...)
transformer.add_adapter(LoraConfig(r=16, lora_alpha=16, ...))
set_peft_model_state_dict(transformer, lora_state_dict)
pipeline = FluxPipeline.from_pretrained(..., transformer=transformer)
image = pipeline(prompt, height=1024, width=1024, ...).images[0]
```

这段实现和论文中的 LoRA/DreamBooth 叙述一致：不是训练一个新的生成模型，而是在 Flux transformer 上挂 LoRA adapter。README 的运行方式也把它作为单独 Stage 1：

```bash
python run_imaginarium_T2I.py \
  --prompt 'A cozy living room featuring comfortable armchairs, a gallery wall, and a stylish coffee table.' \
  --num 4
```

这里有一个重要实现差异：T2I 不在 `ImaginariumPipeline` 内部自动执行。也就是说，I2Layout 接收的是一张已经存在的 image。工程上这很合理，因为用户可以从多张 T2I 候选中挑一张，也可以用 Canny/image editing 得到 guide image，再运行 I2Layout。

### 6.1 为什么 2D guide image 是 domain bridge

如果用户只输入文本，“a cozy living room” 太抽象。LLM 可以列出 sofa、coffee table、armchair，但很难稳定给出艺术上合理的相对位置、朝向、层次和装饰密度。2D guide image 把这些抽象意图落实为视觉结构：

- 哪些物体可见；
- 大物体和小物体的比例；
- 主要视角是正视、斜轴测还是局部特写；
- 哪些物体在墙边；
- 哪些物体互相支撑；
- 哪些物体是重复组，例如多把餐椅；
- 场景整体风格和材质倾向。

后续 parsing、retrieval、pose estimation 都依赖这个图像。如果 T2I 生成的图像不接近资产库，系统就会遇到检索错配；如果图像中大量遮挡或奇异拓扑，pose estimation 就会变得不稳定。因此 Flux 微调不是锦上添花，而是在控制下游误差分布。

## 7. 数据集与资产库

论文强调的一个贡献是高质量资产库和场景布局数据集。根据论文和官方数据页，核心数字是：

| 项目 | 数值 |
| --- | --- |
| 3D scene assets | `2,037` |
| hand-designed scene layouts | `147` |
| asset classes | `500` |
| asset categories | `237` |
| scene types | 论文正文约 `20` 类，附录统计图提到 `21` 类 |
| Imaginarium 平均 objects/scene | `31.86` |
| 3D-FUTURE class count 对比 | `34` |
| 3D-FUTURE 平均 objects/scene 对比 | `5.09` |

这些数字很关键，因为 Imaginarium 不是只靠算法提升。它的系统能力很大一部分来自资产库质量和数据标注。论文说这些资产来自 custom-commissioned models、高质量开源内容和 licensed marketplace items，并由 20 位具有三年以上经验的专业艺术家组织成场景。

官方 Hugging Face 数据集进一步把数据分成几类：

| 数据包 | 用途 |
| --- | --- |
| `imaginarium_3d_scene_layout_dataset_part[1-4].tar.gz` | 完整研究数据，含 `.blend`、RGB、depth、bbox、mask、meta |
| `flux_train_data.tar.gz` | Flux 微调用的 RGB 和 meta |
| `imaginarium_assets.tar.gz` | 推理使用的 3D asset library |
| `imaginarium_assets_internal_placement_space.tar.gz` | 容器内部可放置空间 |
| `imaginarium_asset_info.csv` | 资产元数据 |
| `background_texture_dataset.tar.gz` | 代码后续优化加入的背景贴图数据 |

派生数据集则包含：

| 派生数据 | 用途 |
| --- | --- |
| `imaginarium_assets_render_results_part[1-4].tar.gz` | 多视角 asset render，用于 retrieval / pose |
| `imaginarium_assets_patch_embedding.tar.gz` | DINOv2 patch embeddings |
| `imaginarium_assets_voxels.tar.gz` | S4 collision / optimization 用 voxel |
| `ae_net_pretrained_weights.pth` | AENet pose matching |
| `dinov2_vitl14.pth` | DINOv2 |
| `depth_anything_v2_metric_hypersim_vitl.pth` | Depth Anything V2 |
| `blender-4.3.2-linux-x64.tar.gz` | 预配置 Blender 包 |

**论文-代码对照。** `config/config-example.yaml` 把这些资源路径串起来：

```yaml
fbx_csv_path: asset_data/imaginarium_asset_info.csv
assets_render_result_folder: asset_data/imaginarium_assets_render_results
background_texture_dataset_path: asset_data/background_texture_dataset
ori_dino_weights_path: weights/dinov2_vitl14.pth
S0_geometry_pred.load_from: weights/depth_anything_v2_metric_hypersim_vitl.pth
S3_pose_inference.ae_net_weights_path: weights/ae_net_pretrained_weights.pth
S4_blender_layout_and_corr.precomputed_voxel_dir: asset_data/imaginarium_assets_voxels
```

这说明 Imaginarium 的“模型”不是一个单独 checkpoint，而是由 asset library、render cache、embedding cache、VLM/API 配置、Blender 环境和多个视觉模型共同组成的系统。复现难度也主要来自这里。

## 8. S0 Geometry：Depth Anything V2 与点云构建

论文在 image analysis 中讲到用 Depth Anything V2 估计 depth map，再根据 camera intrinsics 把它转成 point cloud。代码里这个模块独立成 `modules/geometry.py`，对应 `GeometryModule`。

它的主要流程是：

```text
input image
  -> load Depth Anything V2
  -> infer depth
  -> depth.png
  -> estimate focal length from image width
  -> store depth / RGB / intrinsics in Context
```

代码中的焦距估计是一个明显的工程近似：

```python
def estimate_focal_length(width):
    return int(30 / 36 * width)
```

注释写的是 UE5 camera: 30mm focal length, 36mm sensor size。也就是说，它假设输入 guide image 接近某种固定相机设定。这个假设在 pipeline 中很重要，因为 depth-to-point-cloud 需要 camera intrinsics；如果输入图像视角和假设差异很大，后续 OBB 和 translation 都会偏。

`GeometryModule.run()` 还实现了 stage resume：如果 `S0_geometry_pred_results/depth.png` 已存在且不是 clean mode，就不重新跑 depth model，而是加载已有 depth 并把 `depth_image`、`ori_image_numpy`、`intrinsics` 塞进 `Context`。

### 8.1 论文-代码对照

| 论文概念 | 代码实现 | 风险 |
| --- | --- | --- |
| Depth Anything V2 | `DepthAnythingV2` in `modules/geometry.py` | 权重路径缺失直接失败 |
| Depth map $D$ | `depth.png` | 单图深度有尺度和边界误差 |
| Camera intrinsics $K$ | `fx=fy=30/36*width`, `cx=width/2`, `cy=height/2` | 相机假设不一定适合所有输入 |
| Point cloud $P$ | `create_point_cloud` | 前景 mask 错会污染 object point cloud |
| 中间结果传递 | `Context.set_data(...)` | 运行中断后部分数据需从磁盘恢复 |

这个模块看似简单，但它决定了所有几何推断的坐标基础。后面 S1/S2 的 OBB、floor/wall plane、S3 translation、S4 support 都依赖它。论文把它归到 scene image analysis 内，代码把它提前成 S0，是合理的工程拆分。

## 9. S1 Parsing：VLM + Grounding-DINO + SAM + RANSAC

S1 是整个系统中最复杂、最容易被低估的部分。论文说 image parsing module 会恢复视觉语义和几何信息，代码里实际包含一串子任务：

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

代码入口是 `modules/parsing.py` 中的 `SemanticParsingModule.run()`，但真正大部分逻辑在 `modules/_s1_legacy_functions.py`。这不是小细节。官方代码把旧脚本封装成模块，但并没有把 S1 的所有功能拆成干净的新类。报告读代码时必须承认这一点：Imaginarium 是研究系统工程化整理版，不是极简教学代码。

`SemanticParsingModule` 的职责包括：

- 读取 `imaginarium_asset_info.csv`；
- 把 `Context` 中的 GPT/VLM client、字体、Grounding-DINO token 注入 legacy 全局变量；
- 检查 `scene_graph_result.json`、`floor_walls_pose.json`、`final_detect_items.pkl`、`masks.pkl` 是否存在；
- 调用 `run_scene_parsing_pipeline(...)`。

### 9.1 VLM object parsing

论文写的是 GPT-4o；当前 README/config 示例则说明 LLM endpoint 和 model 是可配置的，并提到最近调试使用了 Claude Sonnet 4.5 风格模型名。因此本文不把实现写死成“只支持 GPT-4o”，而写成“论文使用 GPT-4o，代码通过 `utils/llm_api.py` 封装可配置 VLM/LLM API”。

S1 的 object parsing 先让 VLM 在预定义类别范围内识别场景物体。这个约束很重要，因为后续检测、检索和资产匹配都依赖 asset library category。如果 VLM 输出开放词表，例如 “fancy hybrid wardrobe-shelf”，系统仍要映射到最接近的 asset category。

### 9.2 Grounding-DINO + SAM

VLM 给出 object list 后，代码使用 Grounding-DINO 做 grounding detection，再用 SAM 生成 mask。这个组合的好处是把语言类别和像素区域连接起来：

```text
object names
  -> Grounding-DINO boxes
  -> SAM masks
  -> per-object crop / mask / bbox
```

但风险也很明显。小物体可能检测不到，遮挡物体可能 mask 不完整，重复物体可能混淆。论文 Table 3 中 secondary object recovery 只有 `70.41%`，就反映了这类限制：小物体和非结构关键物体的恢复比 primary object 更难。

### 9.3 RANSAC floor/wall extraction

代码中的 `utils/ransac.py` 负责从 depth point cloud 和背景 mask 里估计 floor、wall、ceiling 的 plane。相关函数包括：

- `fit_floor_pcd`
- `fit_walls_pcd_and_pred_xyz`
- `find_floor`
- `find_walls`
- `estimate_floor_and_walls`

这部分对应论文中的 background geometry parsing。它把单图深度里的平面结构变成 scene coordinate system。后续 against wall、hanging on wall、floor support 都需要这些 plane。

### 9.4 Scene graph construction

S1 的输出不只是 masks，也包括 scene graph。legacy 文件中相关函数包括：

- `generate_scene_graph_geometry`
- `verify_floor_parent_with_vlm`
- `supported_generate`
- `against_wall_generate`
- `against_wall_generate_top_down`
- `analyze_groups_and_facing_relations`
- `vis_scene_graph`

这些函数把 VLM 判断、mask overlap、OBB、floor/wall 几何结合起来，生成后续 S4 可用的关系约束。论文里这部分称为 support relationship 和 wall proximity relationship；代码进一步加入 groups 和 facing relationships。

## 10. Prompt 工程精读

`prompts/used_prompts.py` 是理解 Imaginarium 的关键文件之一。因为系统里很多“视觉语义 -> 空间关系”的判断不是由一个端到端模型学出来，而是通过 VLM prompt 工程实现。

### 10.1 Hierarchical traversal prompt

`SCENE_HIERARCHICAL_TRAVERSAL_PROMPT` 要求模型按区域、父物体、子物体三层结构输出 JSON，并强制物体名称来自预定义类别列表。它的核心作用不是简单列物体，而是把场景拆成 hierarchy：

```text
area
  -> parent object
    -> child objects
```

这和论文里的 support tree 有直接关系。场景里很多小物体不应当作为独立 floor-supported object，例如书桌上的书、床上的枕头、柜子里的装饰物。先让 VLM 输出 parent-child 结构，可以减少后续 scene graph 的搜索空间。

### 10.2 Scene graph prompt

`GENERATE_SCENE_GRAPH_PROMPT_S1` 要求 VLM 为每个物体判断：

- `parent`
- `isAgainstWall`
- `directlyFacing`

`GENERATE_SCENE_GRAPH_PROMPT_FLOOR_WALL` 则进一步判断：

- `isAgainstWall`
- `isOnFloor`
- `isHangingFromCeiling`
- `isHangingOnWall`

这些字段几乎就是 S4 layout constraints 的语义来源。比如 `isHangingFromCeiling` 会影响天花板支撑，`isAgainstWall` 会影响 wall alignment，`parent` 会影响 support tree。

### 10.3 Group and facing prompt

`GENERATE_SEMANTIC_RELATIONSHIPS_PROMPT` 的重点是识别视觉上完全相同的 object groups，以及围绕 table 的 chair/stool facing relationships。README 中也说当前代码库增加了 Scene Graph "Groups"：重复视觉特征和相似语义的对象共享资产检索结果，例如所有 dining chairs 匹配同一个 asset。

这其实是一个很有价值的工程补丁。论文原始方法强调 asset retrieval，但如果每把餐椅独立检索，可能得到风格不同的椅子。group 机制把“成套家具”这个设计常识注入系统，提高视觉一致性。

### 10.4 Size correction prompt

S2 中有 VLM dimension refinement。代码 `_s2_legacy_functions.py` 里有 `refine_dimensions_with_vlm` 和 `VLM_SIZE_CORRECTION_PROMPT` 相关逻辑。单图深度估计的 OBB 尺寸经常不准，尤其是遮挡或薄物体。让 VLM 结合图像理解和类别常识修正尺寸，是一个工程上很实用但也有风险的做法：VLM 的尺度判断不是严格几何测量，容易受透视和上下文影响。

## 11. Scene Graph Construction：从图像关系到可执行约束

论文选择了两类关键关系：

1. Support relationship：对象 $a$ 支撑对象 $b$，记作 $\text{obj}_a \prec \text{obj}_b$。
2. Wall proximity relationship：对象与 wall/ceiling 等结构元素接触。

这些关系看起来简单，但足以覆盖大量布局逻辑：

- 书在桌子上；
- 枕头在床上；
- 吊灯挂在天花板；
- 画挂在墙上；
- 柜子靠墙；
- 椅子围绕桌子；
- 小物体放在容器内部空间。

论文把 scene graph construction 分成三步：

| 步骤 | 作用 |
| --- | --- |
| floor support tree | 找出 floor-supported objects，并递归建立 support tree $\mathcal{T}$ |
| ceiling-supported objects | 判断吊灯等天花板支撑对象 |
| objects against walls | 判断哪些对象接触哪些墙 |

附录中还提到，floor support tree analysis 的实验准确率约 `91.95%`。这个数字说明 VLM + 几何约束在结构关系上有一定可靠性，但仍不是完美推理。

**论文-代码对照。** 代码中的 scene graph 不是单一函数做完，而是多个阶段协同：

```text
VLM hierarchical parsing
  -> Grounding-DINO/SAM masks
  -> RANSAC floor/wall planes
  -> scene graph generation prompt
  -> floor parent verification
  -> support / against wall generation
  -> groups and facing relation analysis
```

这里最重要的工程取舍是：不试图让 VLM 一次性输出最终 3D 坐标，而是让它输出更稳定的关系判断。关系判断再被几何模块和 S4 优化器转化为约束。

## 12. S2 Retrieval：资产检索与尺寸修正

S2 的目标是把图像中的每个 object mask 映射到资产库里的具体 3D asset。代码入口是 `RetrievalModule.run()`，主要输出：

- `pcd_obb_data.json`
- `retrieval_results_final.json`
- `texture_retrieval_results.json`

它的执行步骤可以概括为：

```text
S1 masks + depth
  -> compute OBB data
  -> VLM refine dimensions
  -> DINOv2 asset retrieval
  -> optional texture retrieval
  -> store retrieval_results in Context
```

### 12.1 OBB 数据

S2 首先调用 `cal_pcd_obb_data(input_folder, save_folder)`。它基于 S1 mask 和 S0 depth 计算每个物体的点云 OBB。这些 OBB 不只是用于尺寸估计，还会给 S3 pose 和 S4 layout 提供几何先验。

### 12.2 VLM dimension refinement

`refine_dimensions_with_vlm(...)` 对 OBB 尺寸做修正。论文里的 scale estimation 主要讲如何根据 OBB 和资产尺寸做 scaling；代码在 retrieval 阶段就提前加入 VLM 尺寸修正。这是实现上比论文叙述更工程化的地方。

### 12.3 DINOv2 asset retrieval

代码通过 `original_dino_processor` 和 `original_dino_model_for_retrieval` 复用 DINOv2。`_s2_legacy_functions.py` 中的检索逻辑包括：

- 提取 detected object crop 的 image features；
- 加载 asset render embeddings；
- 按类别候选、视觉相似度和 size compatibility 排序；
- 删除或重排某些不一致项；
- 对 group 对象保持检索一致性。

可以把检索分数理解成：

$$
score = \text{visual\_similarity} - \alpha \cdot \text{size\_difference},
$$

其中论文附录提到 $\alpha=0.1$。代码实现中有 `calculate_size_difference` 和 `_process_and_sort_candidates` 一类函数，说明尺寸兼容不是后处理，而是参与候选排序。

### 12.4 Texture retrieval

README 说当前代码库相比原论文增加了 background texture support。`RetrievalModule` 中也可以看到旧的 DINOv2 texture retrieval 被保留为注释，新的流程使用 VLM-based texture retrieval，把 generic category 映射到具体 object instance。这属于官方代码后续优化，文章需要标注它不是论文核心贡献之一，但影响当前实现效果。

### 12.5 检索失败的后果

Asset retrieval 是 Imaginarium 的脆弱点之一。如果检索错，后续 pose estimation 会用错误 asset 做 template matching，S4 也只能优化错误对象的位置。论文附录 failure case 中提到 novel topology mismatch：图像生成了一个混合拓扑的柜子，但资产库没有对应模型，系统只能检索普通 wardrobe，导致子物体放置逻辑失效。

因此，Imaginarium 的生成质量不是只由 Flux 或 Depth Anything 决定，而是由资产库覆盖率、检索特征、类别映射和尺寸估计共同决定。

## 13. S3 Pose：旋转、平移、缩放估计

S3 是论文中技术含量最高的部分之一。代码入口是 `PoseModule.run()`，核心 legacy 函数是 `inference_obj_pose()`，辅助函数来自 `utils/view_matching.py`、`utils/obb.py` 和 AENet 模块。

S3 的输入包括：

- S1 的 `floor_walls_pose.json`；
- S1 的 `scene_graph_result_final.json`；
- S2 的 `retrieval_results_final.json`；
- S2 的 `pcd_obb_data.json`；
- S0 的 `depth.png`；
- asset render templates；
- AENet / DINOv2 权重。

输出是：

```text
S3_pose_inference/{scene_name}_placement_info.json
```

这个 placement info 会被 S4 Blender 脚本加载，成为最终 layout optimization 的输入。

### 13.1 162-view template matching

论文说对每个 asset 从 `162` 个预采样视角渲染 template images，然后提取 pose-sensitive features。代码 `_s3_legacy_functions.py` 中有明显对应：

- `mp_preprocess_template_rgb_and_mask`
- `parallel_process_template_rgb_and_mask`
- `load_features`
- `parallel_extract_template_ae_features`
- `inference_obj_pose`

代码里也能看到形如 `[N, 162, 1024, 16, 16]` 的特征注释，说明每个 candidate asset 会有 162 个 view features。

### 13.2 AENet / local matching

论文引用 GigaPose 的思路，使用 AENet/DINOv2 进行局部特征匹配。代码中 `models/ae_net` 和 `models.ae_net.matching.LocalSimilarity` 对应这一块。它不是直接比较整图 CLIP embedding，而是做更适合 pose matching 的 local correspondence。

### 13.3 Homography reranking

论文说 coarse selection 后，用 homography transformation 和 RANSAC 做 fine selection，抑制对称歧义。代码中对应函数包括：

- `compute_homography`
- `extract_rotation`
- `rotation_difference`
- `homography_rotation_distance`
- `cal_homography_rotation_difference`

它的核心思想是：如果 template view 和 query crop 真的视角一致，那么匹配点之间的 homography 不应该表现出过大的旋转/变形。代码里会对候选 view 重新排序。

### 13.4 OBB geometric enhancement

论文中的 geometric enhancement 使用单图深度得到的 OBB 方向来辅助视角选择。代码中对应：

- `find_view_best_match_obb`
- `convert_obb_pose_to_blender_coordinates`
- `rotation_matrix_to_angle_diff`
- `convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates`

代码里还有小物体 fallback：如果物体太小、OBB 不可靠，就不使用 OBB 方向优化，而直接依赖视觉匹配。这和论文的“OBB guidance unreliable 时退回 visual candidate”一致。

### 13.5 Translation 和 scale

translation 的初始值主要来自 OBB center。代码中会把坐标转换到 Blender 坐标系，例如把深度坐标中的轴映射到 Blender 的 `x, z, -y` 类形式。scale 则需要结合：

- depth-derived OBB size；
- retrieved asset bbox size；
- mask truncation；
- category-specific scaling strategy；
- S4 internal placement 和 collision constraints。

因此，S3 的输出并不是最终 layout，而是一个足够好的初始 placement。真正让它变成可用场景的是 S4。

## 14. S4 Layout：Blender 布局修正与物理仿真

S4 是 Imaginarium 从“看起来像”到“能放得住”的关键。代码入口是 `modules/layout.py`，它不直接在当前 Python 进程里做布局，而是启动 Blender：

```text
blender --background
  --python modules/S4_blender_layout_and_corr.py
  --
  --obj_placement_info_json_path ...
  --output_folder ...
```

这样做有几个原因：

1. 资产加载、材质、渲染和物理仿真都在 Blender 里更自然。
2. Blender 的 Python 环境和主 pipeline 的 PyTorch 环境隔离，降低显存/依赖冲突。
3. S4 很重，单独子进程失败时更容易定位。

`S4_blender_layout_and_corr.py` 是仓库里最大的实现文件之一。它包含：

- `BlenderManager`
- `VoxelManager`
- `ObjManager`
- `RelativePoseManager`
- `layout(...)`
- wall alignment 函数；
- internal subspace 函数；
- collision resolution；
- simulated annealing；
- physics drop simulation。

### 14.1 Local refinement

论文中的 local transformation refinement 在代码里对应：

- `process_rotation_against_wall`
- `process_rotation_against_wall_hierarchical`
- `process_translation_against_wall`
- `process_wall`
- `process_directly_facing`

这类函数把 scene graph 中的 against wall、facing、parent-child 关系转化为对象旋转和平移修正。例如靠墙的柜子需要背面贴墙，吊挂对象需要与天花板接触，某些 face-to-face 关系需要调整朝向。

### 14.2 Internal placement spaces

Imaginarium 数据集为可容纳物体的资产预标注了 internal placement space。代码中相关函数包括：

- `get_closest_subspace`
- `resolve_collisions_in_subspace`
- `align_obj_to_closest_subspace`
- `scale_obj_to_fit_subspace`
- `create_subspace`

这对应论文 Fig. 7 的内部放置逻辑。比如书、花瓶、装饰物可以放进柜子、书架、托盘或桌面局部空间，而不是简单把所有 child object 放在 parent OBB 的顶部。

### 14.3 Voxel collision and simulated annealing

论文中的 global translation optimization 需要减少穿模、保持 support、靠墙和视觉对齐。代码里 `ObjManager` 有：

- `calc_overlap_area`
- `calc_constraints`
- `try_perturb_random_obj`
- `simulated_annealing`

`VoxelManager` 负责把 mesh 或预计算 voxel 加载成可快速碰撞计算的 proxy。论文附录也说使用 voxel representation 来降低 mesh intersection 计算复杂度。

### 14.4 Blender physics

最后，代码调用 `scripts.drop_sim_script.run_drop_simulation` 做物理仿真，并有 `add_rigid_body` 设置 rigid body 参数。论文中说 pillows、stacked objects 等需要真实物理行为，这在代码里是一个明确的 Blender physics stage。

### 14.5 S4 的工程意义

S4 不是简单后处理。它承担了三个任务：

1. 把视觉估计误差修成物理可用布局。
2. 把 scene graph 中的关系转化为 geometry constraints。
3. 把 asset-level 细节，如 internal subspace、voxel、rigid body，纳入最终输出。

没有 S4，Imaginarium 只是一个 image-to-asset-pose 的估计器；有了 S4，它才更接近 production layout system。

## 15. 公式与优化目标精读

Imaginarium 的公式不多，但每个公式都对应一个工程模块。

### 15.1 视觉相似度

论文中的 template matching 相似度可以写成：

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

直觉上，这是比较 asset 在第 $v_k$ 个视角渲染图和 query mask crop 的局部特征匹配。它不只是整图相似度，因为 pose 估计需要找到结构对应点。

代码中这部分分散在 AENet feature extraction、`LocalSimilarity` 和 `inference_obj_pose()` 中。文章不应把它写成一个干净的单函数实现；真实代码是 batch feature、mask crop、template load、view排序、多候选合并的组合。

### 15.2 Homography fine selection

论文中的 fine selection 使用 homography SVD 结果与 identity 的 Frobenius norm：

$$
\{v_i^{vis}\}_{i=1}^{k}
=
\arg\min_{v\in V_{can}}^{(k)}
\left\lVert U_vV_v^T-I \right\rVert_F^2.
$$

它想表达的是：匹配点之间的 homography 越接近稳定几何变换，候选视角越可信。代码中对应 `compute_homography`、`homography_rotation_distance`、`cal_homography_rotation_difference`。

### 15.3 OBB 与视觉候选融合

论文中的最终方向选择逻辑可以概括为：

$$
(v_*^{obb},v_*^{vis})
=
\arg\min
\arccos
\left(
\frac{\text{Trace}(R^{v^{vis}T}R^{v^{obb}})-1}{2}
\right),
$$

并使用阈值：

$$
v_{best}
=
\begin{cases}
v_*^{obb}, & \theta \le \tau, \\
v_1^{vis}, & \theta > \tau.
\end{cases}
$$

论文中 $\tau=\pi/5$。代码中不一定以完全同名变量出现，但逻辑上对应 OBB candidate 与 visual candidate 的角度差、small-object fallback 和 best view selection。

### 15.4 Global post-optimization

论文的 translation optimization 目标是：

$$
\min_{\{t_i^{update}\}}
\sum_i
\lambda_1
\left\lVert t_i-t_i^{update}\right\rVert_2^2
+
\left\lVert
\mathbf{m}_i-
\mathcal{R}_{\mathbf{m}}(\text{obj}_{\mathbf{m}_i},v_{ref})
\right\rVert_2^2.
$$

约束包括：

$$
\text{obj}_{\mathbf{m}_i}\cap \text{obj}_{\mathbf{m}_j}=\emptyset,
\quad
z(\text{obj}_{\mathbf{m}_i})_{max}=t^c,
\quad
d(\text{obj}_{\mathbf{m}_i},\text{obj}_w)=0,
\quad
z(\text{obj}_{\mathbf{m}_j})_{min}=z(\text{obj}_{\mathbf{m}_i})^*.
$$

论文中 $\lambda_1=0.1$。代码里对应的是 `ObjManager.calc_overlap_area()`、`calc_constraints()`、`simulated_annealing()` 以及 wall/support preprocessing。它不一定严格按论文公式写成一个显式优化器，而是用 voxel intersection、constraint penalty 和 simulated annealing 共同实现。

## 16. 实验设置与结果精读

论文实验包括用户研究、专业艺术家评价、重建 fidelity、姿态估计比较、Flux 微调消融、pose 模块消融和 layout refinement 消融。

### 16.1 实现成本

论文给出的运行时间约为 `240` 秒 on single A100：

| 阶段 | 时间 |
| --- | --- |
| text-to-image generation | `10s` |
| scene image analysis | `110s` |
| scene layout reconstruction | `60s` |
| layout refinement | `60s` |

README 则进一步说明：T2I 推荐 A100，I2Layout 可在 RTX 3090 及以上运行。这个差异可以理解为论文实验配置和开源推理建议的不同。完整复现时还要考虑多视角 render、AENet embedding、voxel 预计算等离线成本，其中 multi-view render README 标注可能需要 `1-2 days`。

### 16.2 Table 1：学生偏好研究

论文邀请 `100` 名 20-24 岁 senior art students，对比 HOLODECK、LayoutGPT、DiffuScene、InstructScene。问题有两个：

- Q1：哪个布局更 reasonable and realistic？
- Q2：哪个布局更 coherent and aesthetically pleasing？

Table 1 的偏好率如下：

| Baseline | Reasonable Dining | Reasonable Living | Reasonable Bedroom | Aesthetic Dining | Aesthetic Living | Aesthetic Bedroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffuScene | 75.69 | 82.59 | 79.37 | 74.86 | 85.57 | 80.72 |
| HOLODECK | 79.27 | 77.08 | 76.79 | 82.72 | 72.92 | 74.55 |
| LayoutGPT | N/A | 76.69 | 76.50 | N/A | 77.54 | 81.11 |
| InstructScene | 66.33 | 68.46 | 61.29 | 69.39 | 75.17 | 72.90 |

论文总结平均偏好率：在 reasonableness/realism 上，Imaginarium 相比 DiffuScene、HOLODECK、LayoutGPT、InstructScene 分别为 `79.22%`、`77.71%`、`76.60%`、`65.36%`；在 aesthetic 上分别为 `80.38%`、`76.73%`、`79.33%`、`72.49%`。

这说明 Imaginarium 在主观感知上有明显优势。但需要注意，这类实验受场景选择、资产库公平性、纹理是否移除、参与者背景影响。它强烈支持“更美观、更丰富”，但不是物理正确性或生产效率的唯一证据。

### 16.3 Table 2：专业艺术家和 GPT-4o 评分

论文邀请 `20` 位至少三年经验的专业艺术家，对 `60` 个场景按 composition、semantic logic、aesthetic appeal 打分。Table 2 把人类评分和 GPT-4o 评分写成 `human/GPT-4o`：

| Method | Composition | Semantic | Aesthetic | Overall |
| --- | --- | --- | --- | --- |
| Ours | 3.35 / 3.16 | 3.29 / 2.86 | 3.37 / 3.16 | 3.34 / 3.06 |
| DiffuScene | 2.86 / 3.07 | 2.80 / 2.78 | 2.83 / 3.07 | 2.83 / 2.97 |
| HOLODECK | 2.71 / 2.91 | 2.56 / 2.55 | 2.80 / 2.86 | 2.69 / 2.77 |
| LayoutGPT | 2.42 / 2.97 | 2.26 / 2.83 | 2.35 / 2.97 | 2.34 / 2.92 |
| InstructScene | 2.91 / 3.07 | 2.75 / 2.83 | 2.89 / 3.08 | 2.85 / 2.99 |

这里最值得注意的是：GPT-4o 评分对一些 baseline 也给出接近 3 的分数，而人类评分差异更大。说明自动 VLM 评价可以作为辅助，但不能替代专业评审。

### 16.4 Table 3：重建 fidelity

论文随机选 `30` 个数据集场景，用其 render image 测试系统恢复 ground-truth layout 的能力。结果：

| Metric | Primary | Secondary |
| --- | ---: | ---: |
| Object Recovery | 92.31% | 70.41% |
| Category Preservation | 95.83% | 91.67% |
| Rotation AUC@60° | 74.83% | 71.51% |
| Translation AUC@0.5m | 84.32% | 80.40% |

其他指标包括 scene graph accuracy `93.26%`、CLIP guide image `27.03`、CLIP render image `25.83`、GPT-4o rating `8.29/10`。

Primary 和 secondary 的差距很重要。系统更擅长恢复结构关键物体，例如地面支撑、天花板支撑、靠墙物体；对小物体和装饰物更难。这和 S1 检测、SAM mask、单图深度分辨率的限制一致。

### 16.5 Table 4：旋转估计比较

论文构建了 3DF-CLAPE，包含 category-level 和 instance-level 两个子集。Table 4 的 AUC@60°：

| Method | Category-level | Instance-level |
| --- | ---: | ---: |
| DINOv2 | 31.68% | 31.38% |
| SPARC | 52.54% | 61.46% |
| DiffCAD | 26.45% | 25.44% |
| Orient Anything | 56.07% | 56.24% |
| GigaPose | 39.85% | 57.43% |
| AENet | 45.32% | 62.16% |
| Ours | 70.06% | 81.44% |

这张表证明 Imaginarium 的 pose module 不是简单直接调用 GigaPose。它结合了 template matching、homography reranking 和几何 OBB enhancement，所以在 category-level 和 instance-level 上都明显更强。

### 16.6 Table 5-6：Flux 微调消融

论文用 vanilla Flux 和 finetuned Flux 分别生成 `100` 张 scene images，并人工标注 asset library 中的 ground truth matches。检索精度：

| Metric | Vanilla Flux | Finetuned Flux |
| --- | ---: | ---: |
| Top-1 Accuracy | 48.57% | 68.70% |
| Top-3 Accuracy | 68.57% | 83.21% |

这说明 Flux 微调的核心收益不是“图片更漂亮”，而是让生成图像更像资产库中的东西，从而提升 retrieval 和 pose estimation。

Table 6 则显示过拟合和多样性：

| Model | NN LPIPS up | Scene Sim. to Training down |
| --- | ---: | ---: |
| Vanilla Flux | 0.6375 | 0.3665 |
| Finetuned Flux | 0.5981 | 0.3899 |

| Model | DIV LPIPS up | Intra-set Scene Sim. down |
| --- | ---: | ---: |
| Vanilla Flux | 0.5782 | 0.2974 |
| Finetuned Flux | 0.5901 | 0.3178 |

微调后的 scene similarity 稍高，说明对训练集布局有更强接近性，但并没有严重崩塌。论文的解释是：Flux 更容易学习 style 和 viewpoint，适度学习 object texture/shape，但布局仍保持多样，因为 scene-level multi-body constraints 很复杂且样本每个场景都比较独特。

### 16.7 Table 7：旋转估计消融

| AENet | Homography | Geometry | mAP@5° | mAP@15° | mAP@45° |
| --- | --- | --- | ---: | ---: | ---: |
| yes | no | no | 4.30% | 15.34% | 67.92% |
| yes | yes | no | 5.21% | 59.42% | 76.07% |
| yes | no | yes | 36.22% | 71.73% | 77.16% |
| yes | yes | yes | 66.57% | 75.28% | 80.61% |

这张表很好地解释了为什么 S3 要复杂。AENet 本身能给出粗方向，但 mAP@5° 很低。Homography 对中等/大阈值帮助明显；geometry 对低误差阈值帮助非常大；二者结合效果最好。

### 16.8 Table 8：layout refinement 消融

| Method | Support Correctness | Intersection Pairs | GPT-4o |
| --- | ---: | ---: | ---: |
| Initial Estimation | 62.45 | 5.43 | 2.83 |
| + Local Refinement | 72.86 | 4.43 | 3.07 |
| + Global Optimization | 90.80 | 2.21 | 3.26 |
| + Physical Constraints | 91.34 | 2.20 | 3.29 |

这张表说明 S4 不是可有可无。Global optimization 对 support correctness 和 intersection pairs 的提升最大，physical constraints 主要做最后微调。代码中的 `simulated_annealing`、voxel collision 和 `run_drop_simulation` 正是这个实验背后的工程实现。

## 17. 消融实验与源码模块对应

把论文消融映射到源码，可以得到一张很实用的调试表：

| 消融项 | 论文含义 | 代码位置 | 如果关闭会怎样 |
| --- | --- | --- | --- |
| Flux fine-tune | 资产库风格对齐 | `run_imaginarium_T2I.py`, `scripts/flux/train.py` | 检索 Top-1/Top-3 下降 |
| Homography | 候选 view reranking | `_s3_legacy_functions.py` | 中等角度阈值准确率下降 |
| Geometry | OBB direction enhancement | `utils/view_matching.py`, `utils/obb.py` | 低误差阈值准确率明显下降 |
| Local refinement | wall/support/scale 局部修正 | `S4_blender_layout_and_corr.py` | support correctness 低 |
| Global optimization | collision/support penalty 优化 | `ObjManager.simulated_annealing` | 穿模多、支撑错误多 |
| Physics | Blender rigid body/drop simulation | `drop_sim_script.py` | 软物体/堆叠对象更不自然 |

这张表也能反过来指导复现。如果生成结果差，不能只看最终 render，而要定位是哪一级失败：

```text
T2I 生成内容不贴资产库
  -> S1 检测/分割漏物体
  -> S2 检索错 asset
  -> S3 姿态估计错
  -> S4 约束修不回来
```

## 18. 应用模式：re-editing 与 repainting

论文第 5 节展示了编辑能力。核心思路是：先拿到 3D 场景的渲染图或局部区域，然后在 2D 图像上做局部 repainting，再固定未编辑区域，让系统对新区域重新检索和估计 pose。

可以把它理解成：

```text
已有 3D scene
  -> render global/local image
  -> mask target region
  -> Flux repaint/inpaint
  -> parse edited region
  -> re-retrieve assets
  -> re-estimate pose
  -> merge back into fixed scene
```

论文展示的应用包括：

- global scene completion；
- object replacement；
- local object addition；
- text prompt 控制局部重绘；
- Canny image / guide image 控制。

这里需要谨慎表述。官方 README 展示了 T2I 和 I2Layout 的命令，但不应把论文展示的编辑能力夸大为一个成熟交互式 UI。更稳妥的说法是：Imaginarium 的 pipeline 天然支持这类编辑，因为它把 2D guide image 作为控制接口；具体交互产品还需要额外工程封装。

## 19. 与 CasLayout 的技术对照

由于上一篇精读报告读的是 CasLayout，很自然要对比两者。

| 维度 | CasLayout | Imaginarium |
| --- | --- | --- |
| 核心范式 | 四阶段条件扩散 | 2D guide image + parsing + retrieval + optimization |
| 输入控制 | floor plan、relation graph、text/image 转 relation | text/Canny/image 生成或编辑 guide image |
| 关系建模 | Sparse relation graph + Relation VAE latent | VLM scene graph + support/wall/facing/groups |
| 几何输出 | OBB layout + CAD retrieval | asset instance + rotation/translation/scale + Blender scene |
| 学习模块 | diffusion stages、Relation VAE | Flux LoRA、视觉基础模型、template matching |
| 强项 | 模块化条件生成、关系控制 | 高质量资产复用、视觉丰富度、生产 pipeline |
| 风险 | 级联误差、代码未开放时难复现 | 单图歧义、资产库边界、pipeline 重 |

CasLayout 更像“学习式布局生成系统”：它把 object type、property、relation latent、box layout 都放进扩散框架中。Imaginarium 更像“视觉解析和图形工程系统”：它借助 2D 图像生成模型产生目标，再用检测、分割、深度、检索、姿态匹配和 Blender 优化落地。

两者共同点是都不相信 LLM 直接输出坐标就能解决 3D layout。它们都需要中间结构：CasLayout 用 relation latent，Imaginarium 用 guide image + scene graph + geometry constraints。

## 20. 与相关工作的关系

| 方法 | 主要范式 | 是否使用 2D guide image | 是否依赖 asset library | 是否显式 scene graph | 主要限制 |
| --- | --- | --- | --- | --- | --- |
| ATISS | autoregressive indoor scene synthesis | no | yes | no | 受训练集场景类型限制 |
| DiffuScene | diffusion indoor scene synthesis | no | yes | weak | 数据驱动，资产和场景类型有限 |
| InstructScene | instruction + semantic graph prior | no | yes | yes | 语言到关系仍受限 |
| LayoutGPT | LLM layout planning | no | yes | implicit | 几何精度和稳定性不足 |
| HOLODECK | LLM + 3D scene generation | no | yes | rule-like | 空间约束和复杂资产匹配有限 |
| SceneCraft | LLM writes Blender scripts | no | depends | implicit | 脚本生成稳定性和可控性问题 |
| LayoutVLM | VLM + differentiable optimization | maybe | yes | partial | 评价和适用场景需具体看 |
| CAST | object generation + pose alignment | image/object | less fixed | no | 资产复用和生产属性不足 |
| CasLayout | cascaded diffusion + relation VAE | optional control | CAD retrieval | sparse graph | 依赖模型训练与关系数据 |
| Imaginarium | vision-guided parsing + retrieval + optimization | yes | strong | yes | 单图歧义、资产库边界、pipeline 重 |

Imaginarium 在这个谱系中的定位很明确：它不是最端到端的，也不是最轻量的，而是最强调 **高质量资产库 + 2D 视觉先验 + 工程约束** 的方案。它更接近生产工具链，而不是一个单纯 benchmark 模型。

## 21. 局限性与批判

### 21.1 有限资产库导致 novel topology mismatch

论文附录 failure case 直接指出，如果 image generator 生成了资产库没有的新拓扑，例如 wardrobe 和 bookshelf 的混合体，系统会检索最接近的普通 wardrobe。这样会导致子物体无法合理放置，scene graph 约束也可能失效。

这不是小问题。Imaginarium 的生成能力上限很大程度取决于 asset library 覆盖率。Flux 可以想象任何东西，但 I2Layout 只能摆放已有资产。

### 21.2 单图 pose estimation 本质上有歧义

严重遮挡时，一个物体只露出背面或局部边缘，多个姿态都能产生相似图像。AENet、homography、OBB enhancement 能缓解，但不能彻底解决。

论文也提到未来可以引入 multi-view perspective information。这个方向很合理：如果 T2I 能生成多个一致视角，pose ambiguity 会显著下降。

### 21.3 Flux 多物体一致性仍难

Flux 微调能提升资产库风格对齐，但复杂场景中的多物体一致性仍然困难。2D 图像模型可能生成局部合理但全局不一致的对象组合，例如空间关系不可能、物体尺度不合理、重复对象风格不一致。

### 21.4 多模块串联误差传播

Imaginarium 的 pipeline 很长：

```text
Flux -> VLM -> Grounding-DINO -> SAM -> Depth Anything -> RANSAC -> DINOv2 -> AENet -> OBB -> Blender optimization
```

任一环节错了，后续都可能被带偏。S4 可以修一部分几何错误，但无法把错误类别或错误资产变成正确资产。

### 21.5 依赖外部 API 和权重

代码需要 LLM/VLM endpoint、Grounding-DINO token、Flux 权重、DINOv2、AENet、Depth Anything、Blender、asset data、derived data。复现不是 `pip install && python demo.py` 这么简单。

### 21.6 240 秒级 pipeline 不是实时交互

论文说系统约 240 秒生成场景，相比专业流程 2.5 小时有优势。但如果做实时设计工具，这仍然偏慢。README 也提示第一次运行可能需要等待，派生数据预处理更重。

### 21.7 用户研究和 GPT 评分不能完全代表生产可用性

用户研究说明视觉质量和偏好，但生产可用性还包括资产许可、LOD、碰撞体、动画、交互属性、导出格式、团队工作流、版本管理等。论文和代码开始触及这些问题，但还不是完整 DCC pipeline。

## 22. 复现与工程清单

如果未来要真正复现 Imaginarium，需要按下面检查。

### 22.1 环境

- Python `3.10` conda 环境。
- `pip install -r requirements.txt`。
- Blender `4.3.2`，README 说 `4.0+` 一般支持。
- `config/config.yaml` 从 `config-example.yaml` 复制。
- LLM/VLM endpoint 和 key。
- Grounding-DINO token。

### 22.2 数据

- `imaginarium_assets.tar.gz`
- `imaginarium_assets_internal_placement_space.tar.gz`
- `imaginarium_asset_info.csv`
- `background_texture_dataset.tar.gz`
- optional reference xlsx
- full scene dataset part 1-4，如果做研究评估
- `flux_train_data.tar.gz`，如果重新微调 Flux

### 22.3 派生数据

- asset multi-view render results，README 标注自己生成可能需要 `1-2 days`；
- DINOv2 patch embeddings；
- AENet embeddings，README 标注需本地生成，约 `2 hours`；
- precomputed voxels；
- FBX to Blend optional conversion，约 `20 minutes`。

### 22.4 权重

- `imaginarium_finetuned_flux.pth` 或 README 当前数据中的 LoRA 权重；
- `dinov2_vitl14.pth`；
- `ae_net_pretrained_weights.pth`；
- `depth_anything_v2_metric_hypersim_vitl.pth`。

### 22.5 运行

T2I：

```bash
python run_imaginarium_T2I.py \
  --prompt 'A cozy living room featuring comfortable armchairs, a gallery wall, and a stylish coffee table.' \
  --num 4
```

I2Layout：

```bash
python run_imaginarium_I2Layout.py demo/demo_0.png
python run_imaginarium_I2Layout.py demo/demo_0.png --clean
python run_imaginarium_I2Layout.py demo/demo_0.png --clean --debug
```

### 22.6 输出检查

每个 stage 都有输出目录：

```text
saved_results/{image_name}_result/
  pipeline.log
  stage_logs/
  S0_geometry_pred_results/
  S1_scene_parsing_results/
  S2_3d_retrieval_results/
  S3_pose_inference/
  S4_layout_refinement/
```

要判断运行是否成功，不能只看最后有没有 render。还要检查：

- depth 是否合理；
- masks 是否漏检或错分；
- floor/wall pose 是否稳定；
- scene graph parent/againstWall 是否合理；
- retrieval top candidates 是否贴近图像；
- pose comparison debug image 是否合理；
- S4 overlap、constraints、support correctness 是否改善；
- final render 是否没有明显穿模和悬空。

## 23. 代码阅读总结：这个系统如何被工程化

Imaginarium 的源码设计有几个值得注意的工程点。

第一，`Context` 是全局数据总线。它管理 output directory、logger、device、debug/clean mode、中间数据和 lazy-loaded 模型。这样各 stage 不需要重复加载 DINOv2、AENet 或 GPT client。

第二，stage 支持 resume。S0 检查 `depth.png`，S1 检查 `scene_graph_result.json` 等文件，S2 检查 retrieval outputs，S3 检查 placement info，S4 检查 refined JSON 和 render。对于这种几分钟到数小时的 pipeline，resume 是必需品。

第三，大量复杂逻辑仍在 legacy functions 中。`modules/parsing.py`、`modules/retrieval.py`、`modules/pose.py` 更像 wrapper，把旧脚本封装进新 pipeline。这让代码可运行，但可维护性仍有压力。

第四，Blender 独立进程是合理选择。S4 需要 mesh、材质、voxel、physics 和 render，放在 Blender Python 里比在主 PyTorch 进程里更自然。`LayoutModule` 在运行前释放模型资源，也说明作者考虑了显存压力。

第五，README 当前代码库已经包含论文之外的后续优化，例如 background texture support、scene graph groups、enhanced retrieval。这说明文章不能简单把当前代码等同于论文原版，需要标注“代码库后续优化”。

### 23.1 从 CLI 到 Blender 的调用栈

如果把源码按真实调用栈展开，Imaginarium 的 I2Layout 路径大致是：

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

这条调用栈有两个边界尤其重要。

第一个边界是 `run_imaginarium_I2Layout.py` 和 `pipeline.py` 的边界。CLI 只负责解析输入路径、`--debug`、`--clean`、`--config`，真正的流程编排在 `ImaginariumPipeline.run()`。这意味着如果要把 Imaginarium 集成到 Web 服务或批处理系统，不一定要模拟命令行，可以直接复用 pipeline controller。但由于代码依赖当前工作目录、资产路径和 Blender 命令，工程集成仍然需要封装路径和进程环境。

第二个边界是 `LayoutModule.run()` 和 Blender script 的边界。S4 不是在同一个 Python 进程里直接 import Blender API，而是启动：

```text
blender --background --python modules/S4_blender_layout_and_corr.py -- ...
```

这个选择使主进程和 DCC/physics runtime 解耦。好处是 Blender 依赖、mesh 加载、材质、碰撞和渲染都留在 Blender 环境里；坏处是调试时要同时看 pipeline log 和 Blender stdout，且错误栈可能跨进程断开。

**论文-代码对照。** 论文把 layout refinement 讲成方法模块，代码把它实现成独立 runtime。这是阅读论文时容易忽略的工程现实：3D scene synthesis 里的“物理优化”不是一个纯 PyTorch 后处理函数，它往往需要真实图形软件、mesh 数据结构和渲染器参与。

### 23.2 Context 数据总线：为什么不是简单函数参数

`core/context.py` 是这套代码的关键。它做了四类事情：

| Context 职责 | 代码表现 | 为什么重要 |
| --- | --- | --- |
| 输出目录管理 | 根据 image name 建立 `{image}_result` | 保证 resume 和 stage 输出路径一致 |
| 日志管理 | `pipeline.log` + `stage_logs/` | 多分钟 pipeline 需要按阶段定位问题 |
| 中间数据管理 | `set_data()` / `get_data()` | 在 S0/S1/S2/S3 间传 depth、intrinsics、retrieval result、placement path |
| 模型生命周期 | lazy loading DINOv2/AENet/GPT，必要时 release | 降低启动成本，避免 S4 前显存占满 |

如果把每个 stage 都写成独立脚本，理论上也能运行，但会出现三个问题：重复加载大模型、跨脚本传参繁琐、无法统一控制 clean/resume/debug。`Context` 的作用就是把这些系统性问题集中处理。

这里也能看到研究代码工程化的折中：`Context` 让 pipeline 容易跑通，但也会让模块之间存在隐式依赖。例如 S3 可能从内存里读 `retrieval_results`，也可能从 `retrieval_results_final.json` 回读；S4 可能从 `Context` 读 `placement_info_path`，也可能按默认目录拼路径。这种“双通道”对 resume 友好，但阅读者需要知道实际数据来自哪里。

### 23.3 wrapper 层和 legacy function 层的分工

官方代码里 `modules/parsing.py`、`modules/retrieval.py`、`modules/pose.py` 并不是全部算法细节所在。它们更像 stage wrapper，负责：

- 准备输入输出目录；
- 检查已有文件，决定是否跳过；
- 从 `Context` 获取共享模型或中间数据；
- 调用 `_s1_legacy_functions.py`、`_s2_legacy_functions.py`、`_s3_legacy_functions.py`；
- 把结果写回文件和 `Context`。

真正复杂的视觉解析、VLM prompt 调用、关系图构建、OBB 计算、pose matching 细节，大量仍在 legacy functions 中。这对读者有两个启示。

第一，如果只读 `modules/*.py`，会以为算法很薄；但那只是 wrapper。论文方法的许多细节要继续往 legacy 文件里追。

第二，如果想做二次开发，最稳妥的切入点不是一开始重写 legacy 函数，而是先在 wrapper 层增加检查、缓存、指标和可视化。例如在 S2 后检查 retrieval top-k 语义一致性，在 S3 后检查 rotation confidence，在 S4 前检查 support tree 是否闭环。这些增强对系统稳定性有帮助，也不容易破坏核心逻辑。

### 23.4 论文模块到源码函数的细粒度索引

下面这张表更适合打开源码时对照使用。它不覆盖所有函数，而是列出读论文时最容易追问“代码在哪里”的位置。

| 论文概念 | 代码位置 | 读代码时看什么 |
| --- | --- | --- |
| fine-tuned Flux guide image | `run_imaginarium_T2I.py` | base model、LoRA adapter、resolution、sampling steps、输出目录 |
| Depth + camera intrinsics | `modules/geometry.py` | `estimate_focal_length()`、Depth Anything V2 加载、`depth.png` 保存 |
| point cloud construction | `modules/geometry.py` | `create_point_cloud()` 的 pinhole projection 和 mm/meter 转换 |
| VLM object parsing | `modules/parsing.py`、`modules/_s1_legacy_functions.py` | object list、secondary inspection、dedup |
| Grounding-DINO + SAM masks | `modules/_s1_legacy_functions.py` | bbox/mask 生成和 mask 文件命名 |
| floor/wall RANSAC | `utils/ransac.py`、S1 legacy | 平面拟合、floor/wall pose 输出 |
| scene graph | S1 legacy、`prompts/used_prompts.py` | support、against wall、facing、group relation |
| OBB estimation | `modules/_s2_legacy_functions.py`、`utils/obb.py` | depth/mask/point cloud 如何变成对象 OBB |
| VLM dimension refinement | S2 legacy、prompts | 尺寸异常如何被语言模型修正 |
| DINOv2 asset retrieval | `modules/retrieval.py`、S2 legacy | patch embedding、local/global matching、size compatibility |
| background texture retrieval | `TextureRetrieval` | patch feature、color histogram、wall/floor/ceiling texture |
| 162-view pose matching | `modules/_s3_legacy_functions.py`、`utils/view_matching.py` | template view、AENet、DINOv2 local matching |
| homography reranking | `utils/view_matching.py`、S3 legacy | 局部匹配如何影响旋转候选排序 |
| placement info fusion | `PoseModule.run()`、S3 legacy | pose、retrieval、scene graph、truncation 如何合并 |
| Blender physical refinement | `modules/layout.py`、`modules/S4_blender_layout_and_corr.py` | 子进程调用、wall alignment、support/collision/physics |

这张索引也揭示了 Imaginarium 和端到端生成模型的差异：它的“模型能力”分布在多个 pretrained model、prompt、资产索引、几何算法和物理仿真中。论文精读如果只看神经网络模块，会漏掉很多决定结果质量的工程因素。

### 23.5 错误传播链：从最终 render 倒查问题

对 Imaginarium 这类串联系统，最实用的分析方式是从最终问题倒查。

如果最终 render 里某个重要物体缺失，优先看 S1。缺失通常不是 S4 删除了对象，而是 object parsing 或 grounding 阶段没有稳定识别。检查顺序应该是：guide image 是否清楚、VLM object list 是否包含该物体、Grounding-DINO bbox 是否覆盖、SAM mask 是否有效、`scene_graph_result_final.json` 是否保留。

如果物体类别对但外观明显不对，优先看 S2。可能是 crop 里遮挡太强，DINOv2 local/global feature 匹配到了错误资产；也可能是 asset metadata 的 category 粒度太粗，例如 “chair” 内部包含餐椅、扶手椅、办公椅。此时 S3/S4 只能摆放错资产，无法让错资产变成对资产。

如果外观对但朝向错，优先看 S3。单张图像的旋转估计天然有歧义，尤其是对称物体、遮挡物体、小物体和无明显纹理的物体。AENet/template matching、homography reranking 和 OBB orientation candidate 是三个互补信号，任何一个信号被遮挡或对称性干扰，都可能选错旋转。

如果物体位置大体对但有穿模、悬空或离墙不自然，优先看 S4。S4 负责把图像解析结果转成物理上更合理的 3D layout，但它的目标不是严格重建图像像素，而是在 scene graph、support tree、collision 和 physics 之间折中。某些情况下，S4 会为了消除碰撞而牺牲一点视觉对齐。

如果整个房间尺度不对或墙地关系错，优先回到 S0/S1。Depth Anything V2 提供的是单目深度估计，焦距估计也是近似。floor/wall plane 一旦错，后面所有 translation、scale、wall alignment 都会继承这个全局坐标系错误。

### 23.6 代码级复现检查点

真正跑 Imaginarium 时，建议按下面的 gate 检查，而不是等最终图片生成后再看。

| Gate | 检查对象 | 通过标准 | 不通过时先改哪里 |
| --- | --- | --- | --- |
| G0 | input image | 1024 左右、视角清楚、主要对象可见 | T2I prompt 或手选更好的 guide image |
| G1 | `depth.png` | 地面、墙面、前景对象有相对合理深度层次 | Depth 权重、输入图像、焦距假设 |
| G2 | object masks | 主要家具不漏、不大面积混合背景 | VLM object list、Grounding-DINO threshold、SAM mask |
| G3 | floor/wall pose | 地面水平、墙面方向与图像一致 | RANSAC 参数、点云质量、mask 排除 |
| G4 | scene graph | support parent、against wall、facing/group 大体合理 | prompt、VLM 二次检查、人工修 JSON |
| G5 | retrieval top-k | 类别、风格、尺寸都相近 | asset metadata、DINOv2 features、size refinement |
| G6 | pose debug image | template view 与 crop 匹配，旋转无明显反向 | homography、template view、small-object fallback |
| G7 | S4 render | 无明显穿模/悬空，墙地贴图和对象关系合理 | support tree、voxel、internal placement space、Blender 约束 |

这套 gate 的意义是把论文里的“高质量生成”拆成可操作的工程判断。对于研究复现，能定位到哪个 gate 失败，比单纯报告最终 FID 或用户偏好更有用。

### 23.7 与论文叙述不完全一致的地方

最后，需要明确当前公开代码和论文叙述之间的差异。

第一，论文方法总览更像三大块：prompt expander、scene image analysis、scene layout reconstruction/refinement；代码实际是 T2I 脚本加 S0-S4 五阶段。二者不是矛盾，而是论文抽象和工程执行粒度不同。

第二，论文中强调 GPT-4o 参与视觉语言推理，但当前 `config-example.yaml` 把 LLM/VLM 接口做成可配置形式，README 还提到近期调试使用过 Claude Sonnet 4.5 风格的模型配置。因此，报告里更准确的说法是“依赖可配置 VLM/LLM 接口”，而不是“固定依赖 GPT-4o”。

第三，README 明确列出一些代码库后续优化，例如 background texture support、scene graph groups、local/global asset retrieval。它们增强了当前实现，但不应全部倒推成论文原始实验设置。写论文精读时需要把“论文方法”和“当前官方仓库实现”分开。

第四，代码的可复现性依赖大量外部资产、权重和预计算数据。即使主代码公开，也不等于读者能马上重现实验表格。资产授权、数据包版本、VLM endpoint、Grounding-DINO API、Blender 环境和 GPU 显存都会影响结果。

第五，S4 的许多效果来自图形和物理启发式。论文读者如果只用深度学习评价框架理解它，会低估 Blender layout script 的重要性；工程读者如果只看 Blender script，又会低估前面 parsing/retrieval/pose 的约束作用。Imaginarium 的质量来自这两条线的组合。

## 24. 推荐阅读路径

如果只想快速理解论文，建议这样读：

1. 先读 Abstract 和 Introduction，理解为什么要用 2D guide image。
2. 看 Fig. 2 和 Method Overview，建立 pipeline 全局图。
3. 读 3.1 Prompt Expander，理解 Flux 微调和资产库对齐。
4. 读 3.2 Scene Image Analysis，理解 semantic、geometry、scene graph 三类解析。
5. 读 3.3 Transformation Estimation，重点看 rotation matching、homography、OBB enhancement。
6. 读 3.4 Layout Refinement，理解 S4 为什么必要。
7. 读 Table 1-8，区分用户偏好、重建 fidelity、pose 指标和消融。
8. 最后读 Appendix A.1、A.3、A.4，补充工程细节、数据集和 failure cases。

如果要读代码，建议顺序是：

1. `README.md`：先理解安装、数据、两阶段运行方式。
2. `config/config-example.yaml`：看资源路径和模型依赖。
3. `run_imaginarium_T2I.py`：看 Flux LoRA inference。
4. `run_imaginarium_I2Layout.py`：看 CLI、debug、clean、config。
5. `pipeline.py`：看 S0-S4 stage 顺序。
6. `core/context.py`：看数据总线、lazy models、日志、resume。
7. `modules/geometry.py` 到 `modules/layout.py`：看 wrapper 层。
8. `_s1/_s2/_s3_legacy_functions.py`：看真正复杂逻辑。
9. `modules/S4_blender_layout_and_corr.py`：看 Blender layout 和物理优化。
10. `prompts/used_prompts.py`：看 VLM 关系判断如何被 prompt 化。

## 25. 结论

Imaginarium 的贡献不是“又一个 text-to-3D 方法”。更准确地说，它提出了一个把 2D 视觉生成先验转成 3D 资产布局的系统工程：

```text
高质量 asset library
+ fine-tuned Flux guide image
+ VLM semantic parsing
+ Grounding-DINO / SAM masks
+ Depth Anything / RANSAC geometry
+ DINOv2 / AENet retrieval and pose matching
+ scene graph constraints
+ voxel optimization
+ Blender physics
```

它的核心价值在三个地方。

第一，它利用了 2D 图像模型强大的构图和审美能力，同时没有放弃 3D asset 的可控性和可复用性。

第二，它把复杂场景生成拆成一串可诊断的工程模块。每个模块都有中间文件、debug artifact 和可替换空间。

第三，它真正面对了生产系统的问题：资产库、材质、内部空间、物理仿真、运行缓存、Blender 集成、API 配置和复现成本。

它的核心风险也同样清楚：有限资产库限制开放世界能力；单图姿态估计在遮挡下不适定；VLM、检测、分割、深度、检索、优化的误差会串联传播；完整运行依赖大量数据、权重和工程环境；用户研究和自动评分不能完全代表生产可用性。

从研究角度看，Imaginarium 最值得借鉴的是一个方向判断：当 3D 数据稀缺而 2D 图像模型强大时，不一定要强行训练端到端 3D generator。把 2D guide image 当作中间控制面，再通过视觉解析、资产检索和约束优化落地到 3D，可能是更可控、更贴近生产的一条路线。

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
