# 邻域一致的动态线性分类头

## 代码版本与恢复

G5稳定版本：`dino-baseline-g5`，提交 `790f160689976522c9fd50a0efabecdcd880940a`。
G5服务器目录 `/root/PPM_CLIP_DINO_baseline` 保持不变。
新分支 `experiment/dynamic-classifier`，新目录 `/root/DINOv2_AIGI_dynamic_head`。
结果位于数据盘 `/root/autodl-tmp/outputs/dinov2_vitl14_baseline/dynamic_head_20260905`。
不用破坏性reset：需要旧模型时直接使用G5目录、G5配置及原checkpoint。

## 分类头

冻结DINOv2 ViT-L/14输出1024维CLS特征z，没有文本编码器。
保留G5的两类线性层；等价标量参数w0=w_fake-w_real，b0=b_fake-b_real。

    a(z) = tanh(V normalize(z))
    delta_w(z) = U a(z)
    w(z) = w0 + delta_w(z)
    s(z) = w(z)^T z + b0

V形状16×1024，U形状1024×16。16是生成器秩，不是子空间或类别数量。
基础线性层和U/V都训练，DINO不训练。新增32768个参数，总训练参数34818。
代码保留原来两类logits，并分别加上-residual/2和+residual/2，因而fake-real的差就是s(z)。
U初始化为0、V随机初始化，初始logits逐元素等于G5，不能把U和V都初始化为0。
首个反向步骤V梯度为0是预期现象；U更新后V开始获得梯度。
推理只用原图特征，不采样扰动、不计算一致性损失。
这是低秩乘性交互分类器，输出对z总体非线性；不能把生成的w(z)自动等同于真实梯度。

## 邻域一致性

只在训练的特征空间中，采样随机方向epsilon，令其L2长度=0.01×||z||。
它不是图像新增强，也不需要多跑一次DINO。

    Llocal = mean((s(z+epsilon)-s(z)-w(z)^T epsilon)^2 / (||epsilon||^2+1e-8))
    L = CrossEntropy + lambda * Llocal

实现使用严格等价的稳定公式：

    residual = (delta_w(z+epsilon)-delta_w(z))^T (z+epsilon)

基础w0/b0在此损失中代数抵消，两侧delta_w都保留梯度，不使用stop-gradient。
所有新实验分类头使用FP32；DINO保持原AMP，避免小扰动被FP16舍入吞掉。
噪声使用独立随机数生成器，新头初始化不推进全局随机数状态。
默认实验lambda=1，半径0.01，在测试前预先确定；这不是已经调优的参数。
不会依据GenImage跨生成器或Chameleon最终测试成绩自动选择这些超参数。

## 实验安排

| 名称 | 分类头 | 一致性权重 | 可训练参数 |
|---|---|---:|---:|
| D1_dynamic_local | 动态线性头 | 1 | 34818 |
| C1_linear_continue | G5原线性头，继续训练对照 | 0 | 2050 |
| A1_dynamic_no_local | 动态线性头，消融 | 0 | 34818 |
| C2_residual_mlp | 残差MLP，32维tanh隐层 | 0 | 34850 |

MLP使用相同归一化输入、G5基础线性层与零输出初始化；新增32800个参数，
比动态头多32个，是相近容量的非线性对照，不是严格相同函数类。

四组均从G5最佳checkpoint初始化模型权重，优化器和调度器重新初始化。
每组最多额外训练10轮，保留patience=3早停；不是重新从DINO预训练权重训练。
batch_size=48，accumulation_steps=2，有效batch=96，head_lr=3e-4，dropout=0。
AdamW，weight_decay=0.05，seed=1029，其他参数见run_config.json。
G5历史结果只用于参考；C1控制额外训练和分类头精度变化，避免把它们误认为新模块收益。
这是单种子探索实验，不能凭一次结果声称显著提升。

数据代码不修改：训练=尺寸检查+RandomZoomCrop(p=0.5,scale=2)+随机翻转+归一化；
验证、测试=相同尺寸检查+CenterCrop(224)+归一化，无随机增强。
保留用户确认的val>=0.90后用SDv1.4测试准确率选择checkpoint规则。
G5的SDv1.4验证/选择测试来自同一val目录，仅加载器不同；不是独立测试集。
这一已保留的选择协议需要在论文/报告中如实说明，不能称为完全未触碰的测试集选择。
已知训练集空文件仍沿用原loader返回零张量的处理，不改变数据范围。

每组训练结束使用该组最佳checkpoint测试GenImage八子集和Chameleon。
记录Acc/AUC/AP，GenImage均值是八子集宏平均，Chameleon使用完整现有测试目录。
不改变阈值，不用Chameleon训练或挑checkpoint。

## 文件和运行

- `networks/dynamic_head.py`：动态头、MLP对照、特征噪声与一致性损失。
- `networks/dino_baseline.py`：头类型切换、严格G5初始化、FP32头前向。
- `options.py`：新参数；默认仍为原linear，默认不启用一致性。
- `main.py`：CE和一致性联合训练、分项日志、最佳epoch与运行元数据。
- `data_loading.py`、`validation.py`、`test.py`：原文件不改，评估通过统一build_model选择头。
- `tests/test_dynamic_head.py`：公式/梯度/初始化/RNG/参数量/状态恢复单元检查。
- `scripts/verify_dynamic_head.py`：真实训练图片的GPU前向、反向和冻结检查。
- `scripts/run_dynamic_head_suite.sh`：依次运行四组并完成两个benchmark测试。
- `scripts/summarize_dynamic_head.py`：核对所有结果，生成RESULTS.md/RESULTS.json。

服务器启动：

    cd /root/DINOv2_AIGI_dynamic_head
    bash scripts/run_dynamic_head_suite.sh

脚本使用文件锁防止重复实验，拒绝在未提交代码或混合版本下启动。
已完成训练会跳过，失败的测试可安全重跑；中断训练会明确阻塞，不会静默重新训练覆盖结果。
没有声称可精确恢复优化器/数据加载器中途状态；这种恢复需单独诊断和授权。
每次最佳checkpoint有best_checkpoint.json记录epoch，训练完成有training_complete.json。
