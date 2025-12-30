# DCCF 复现项目 | Disentangled Contrastive Collaborative Filtering

## 1.对比实验

位于复现代码/rechoruscode

## 2.消融实验、超参实验

用原始的DCCF进行修改

修改内容如下：

1. 创建了新的消融类model_for_ablation.py
2. 修改了DCCF_PyTorch.py和praser.py使得可以用参数选择消融类型和输出文件名。
3. 创建了两个文件。run_ablation.py和run_n_intents.py，分别用于消融和对比意图数的影响。
4. log里面是运行出来的结果，results里面是result.ipynb解析log运行出的结果。
5. 修改代码使得可以在CPU上运行

## 怎么运行

### 环境配置



