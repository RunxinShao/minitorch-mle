# MiniTorch：从零手写的深度学习框架

这是我在 2024 年秋季 Cornell Tech《Machine Learning Engineering》课上完成的 [MiniTorch](https://minitorch.github.io/) 全部五个模块。MiniTorch 用纯 Python 从零实现一个迷你 PyTorch：从标量的自动微分开始，一路做到张量、numba 多核并行、CUDA GPU 加速，最后用它训练卷积神经网络识别 MNIST 手写数字和做句子情感分类。

| 目录 | 模块 | 实现的内容 | 官方讲义 |
| --- | --- | --- | --- |
| [`mod0/`](mod0) | Fundamentals | 基础算子、map/zipWith/reduce、`Module` 参数树、hypothesis 性质测试 | [module0](https://minitorch.github.io/module0/module0/) |
| [`mod1/`](mod1) | Autodiff | 中心差分、`Scalar` 计算图、链式法则、拓扑排序、反向传播 | [module1](https://minitorch.github.io/module1/module1/) |
| [`mod2/`](mod2) | Tensors | shape/strides 存储、广播、张量 map/zip/reduce、张量自动微分 | [module2](https://minitorch.github.io/module2/module2/) |
| [`mod3/`](mod3) | Efficiency | numba `prange` 并行、批量矩阵乘法、CUDA 核函数、共享内存分块矩阵乘法 | [module3](https://minitorch.github.io/module3/module3/) |
| [`mod4/`](mod4) | Networks | 1D/2D 卷积、平均/最大池化、softmax/logsoftmax、dropout、MNIST 与情感分类 | [module4](https://minitorch.github.io/module4/module4/) |

每个模块都是一个独立的 Python 包（包名都叫 `minitorch`），后一个模块包含前一个模块的全部代码，所以 `mod4/` 就是完整的框架。

## 环境要求

Python 3.11 或 3.12。CPU 部分在 Windows、macOS、Linux 上都能跑；CUDA 部分（Task 3.3、3.4）需要 NVIDIA 显卡。

## 安装

每个模块单独建一个虚拟环境。下面以 `mod4` 为例，其他模块把目录名换掉即可：

```bash
cd mod4
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements.extra.txt
pip install -e .
```

`requirements.txt` 是跑测试必需的包；`requirements.extra.txt` 是可视化界面和训练脚本用的包（streamlit、plotly、python-mnist、datasets、embeddings 等），只跑测试可以不装。两个文件要在同一条命令里装：单独装 extras 时 pip 会把 NumPy 升到 2.0 以上，而 numba 0.60 只支持到 NumPy 2.0。

## 跑测试

测试按作业任务打了 pytest 标记，和当年 GitHub Classroom 自动评分跑的命令一样：

```bash
pytest                                     # 当前模块的全部测试
pytest -m task2_3                          # 只跑某一个任务
pytest -m "not task3_3 and not task3_4"    # 没有 NVIDIA 显卡时跳过 CUDA 任务
```

| 模块 | 任务标记 | 本地结果 |
| --- | --- | --- |
| mod0 | `task0_1` ~ `task0_4` | 43 passed, 1 xfailed |
| mod1 | `task1_1` ~ `task1_4` | 93 passed, 1 xfailed |
| mod2 | `task2_1` ~ `task2_4` | 65 passed, 3 xfailed |
| mod3 | `task3_1` ~ `task3_4` | 53 passed（CPU）+ 64 passed（CUDA） |
| mod4 | `task4_1` ~ `task4_4` | 66 passed（CPU）+ 64 passed（CUDA） |

`xfailed` 是课程模板里标记为「预期失败」的用例，不是错误。

代码风格检查（当年自动评分的 Style 一项）：

```bash
pip install ruff pyright
ruff check . && ruff format --check .
pyright
```

## 跑训练

所有训练脚本都要在模块根目录下运行。

| 模块 | 命令 | 说明 |
| --- | --- | --- |
| mod1 | `python project/run_scalar.py` | 用 `Scalar` 搭的小网络在二维点集上做二分类 |
| mod2 | `python project/run_tensor.py` | 同一个任务，换成张量实现 |
| mod3 | `python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05` | numba 多核后端；`--BACKEND gpu` 换成 CUDA；`--DATASET` 可选 `simple`、`split`、`xor` |
| mod4 | `python project/run_mnist_multiclass.py` | 卷积网络识别 MNIST；脚本只用训练集，训练集文件已经在 `mod4/project/data/` |
| mod4 | `python project/run_sentiment.py` | 1D 卷积做 SST-2 情感分类，首次运行会下载 GloVe 词向量和 GLUE 数据集，体积较大 |

可视化界面（需要 `requirements.extra.txt`）：

```bash
streamlit run project/app.py -- 3     # 最后的数字是模块编号
```

## CUDA 说明

`requirements.txt` 固定的是 numba 0.60。它自带的 CUDA 支持在较新的 NVIDIA 驱动上（例如 580 版、CUDA 13）会在启动核函数时报 `access violation`。遇到这种情况，换成 NVIDIA 官方维护的 numba-cuda。它会把 numba 和 NumPy 升到和 `requirements.txt` 不同的版本，所以建议单独建一个虚拟环境：

```bash
cd mod3
python -m venv .venv-cuda
source .venv-cuda/bin/activate   # Windows: .venv-cuda\Scripts\activate
pip install "numba==0.61.2" "numpy==2.0.2" "numba-cuda[cu13]" hypothesis==6.54 pytest typing_extensions
# 驱动是 CUDA 12 时把 [cu13] 换成 [cu12]
pytest -m "task3_3 or task3_4"
```

本仓库的 CUDA 测试就是这样在 RTX 4060 Laptop GPU（驱动 580.88）上跑通的。没有本地显卡的话，也可以在 Google Colab 里选 GPU 运行时，clone 本仓库后执行同样的命令。

## 相对当年提交的修正

当年提交的代码通过了自动评分里的功能测试，但训练脚本和几处没有测试覆盖的地方有问题。整理这个仓库时做了下面这些修正。修正后上表的测试、ruff、pyright 全部通过；mod1~3 的训练脚本和 MNIST 训练都实际跑过并且在学习，情感分类脚本能加载真实的 SST-2 数据集，模型部分用随机词向量代替 GloVe 跑通了训练和验证。

| 位置 | 问题 | 修正 |
| --- | --- | --- |
| `mod1/project/run_scalar.py` | `Linear.forward` 用 `.data` 取出浮点数计算后再包成新的 `Scalar`，计算图被切断，梯度传不到权重，网络无法训练 | 换成 mod2 中直接对 `Scalar` 做加法和乘法的版本，保留计算图 |
| `mod2/project/run_scalar.py` | 读取 `data.x[i]`，而数据集的属性名是 `X`，脚本一运行就报错 | 改回 `data.X[i]` |
| `mod1/minitorch/autodiff.py` | `Variable` 协议的方法只有文档字符串、没有 `...`，pyright 报 5 个错误 | 补上 `...` |
| `mod1/tests/test_operators.py` | 没有从 mod0 同步，仍是占位符，6 个测试失败 | 同步 mod0 中写好的版本 |
| `mod2~4/minitorch/tensor.py` | 没有 `__gt__`，`tensor > 0.5` 报 `TypeError` | 接上已实现的 `GT` 函数 |
| `mod2~4/minitorch/tensor_data.py` | 0 维张量（`minitorch.tensor(0.5)`）的 shape/strides 数组被 NumPy 推断成 float64，numba 后端无法用它做下标 | 显式使用模板类型注解里声明的 `int32` |
| `mod2~4/project/run_tensor.py` | `view(-1, ...)` 不被支持；`default_log_fn` 里引用了不存在的 `self` | 写出明确的维度；删去错误的日志行 |
| `mod2~4` 的训练脚本 | 用 `> minitorch.tensor(0.5)` 绕过缺失的 `__gt__`，在 numba 后端上会触发 0 维张量的问题 | 恢复模板原来的 `> 0.5` |
| `mod3~4/project/run_scalar.py`、`run_tensor.py` | 仍是课程模板的占位符，没有从上一个模块同步 | 同步已修正的实现 |
| `mod4/minitorch/cuda_ops.py` | CUDA `tensor_reduce` 没有写完（只有 `pass`），只是 mod4 的测试没有用到 GPU | 同步 mod3 中经过测试的实现 |
| `mod4/minitorch/nn.py` | `softmax` 直接对输入取 `exp`，输入较大时溢出；缺少 Task 4.4 要求的 `argmax` | 先减去最大值再取 `exp`；补上 `argmax` 并在 `Max.backward` 中使用 |
| `mod4/project/run_mnist_multiclass.py`、`run_sentiment.py` | dropout 的开关写成 `self.mode == "eval"`，而 `Module` 没有 `mode` 属性，评估时 dropout 依旧生效 | 改为 `not self.training` |
| `mod4/requirements.extra.txt` | `datasets==1.15.1` 在 Python 3.12 上一 import 就报 `AttributeError: readonly attribute` | 升到 `datasets==3.0.1`（和 2025 版课程一致） |
| `mod4/project/data/` | 测试集标签文件 `t10k-labels-idx1-ubyte` 实际上是测试集图像文件的副本 | 训练脚本只用训练集，仓库里只保留两个训练集文件 |

各模块 README 中的训练日志和截图是当年用修正前的代码跑出来的。

## 致谢

MiniTorch 由 [Sasha Rush](https://github.com/srush) 设计，课程模板来自 [minitorch/minitorch](https://github.com/minitorch/minitorch) 和 Cornell Tech MLE 课程。模板代码的版权归原作者所有，本仓库只用于学习和展示。
