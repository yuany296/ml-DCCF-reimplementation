# DCCF 复现项目 | Disentangled Contrastive Collaborative Filtering

## 1.对比实验

位于复现代码/rechoruscode

### 说明
这是本小组基于论文和ReChorus框架对DCCF模型进行的复现和对比实验。

**实验模型**：BPRMF、LightGCN、DCCF

**数据集**：Grocery_and_Gourmet_Food、MovieLens_1M

结果位于log.

### 怎么运行

#### 环境配置

环境位于requirements.txt

额外用到DCCF的环境
- torch-scatter == 2.0.9
- torch-sparse == 0.6.14

#### 怎么运行

在复现代码/rechoruscode的根目录下运行
```bash
python run_comparison.py
```
要修改参数，只需修改run_comparison.py中的model_params

## 2.消融实验、超参实验

位于复现代码/DCCF

用原始的DCCF进行修改

修改内容如下：

1. 创建了新的消融类model_for_ablation.py
2. 修改了DCCF_PyTorch.py和praser.py使得可以用参数选择消融类型和输出文件名。
3. 创建了两个文件。run_ablation.py和run_n_intents.py，分别用于消融和对比意图数的影响。
4. log里面是运行出来的结果，results里面是result.ipynb解析log运行出的结果。
5. 修改代码使得可以在CPU上运行

### 怎么运行

#### 环境配置
本实验运行时用到的cuda版本为11.3.其他环境与DCCF一致。

#### 运行消融实验脚本
```bash
python run_ablation.py
```
#### 运行意图数超参实验脚本
```bash
python run_n_intents.py
```

## 原始代码链接
[DCCF](https://github.com/HKUDS/DCCF)
[Rechorus](https://github.com/THUwangcy/ReChorus)

