# FT-VLLM
Fine-tuning Scripts for Vision-Language Large Models
## 训练
### 1. 准备数据
例如coco数据集按以下格式方式存放
```plaintext
├── coco
│   └── train2017
```
接着在data/__init__.py中注册数据集，如下所示  
```python
# 定义数据集
MY_DATASET = {"annotation_path": "to/path/json",
    "data_path": "to/path",}
data_dict = {"mydataset": MY_DATASET}
```
由于目前部分代码比如LazySupervisedDataset和DataCollatorForSupervisedDataset是基于[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) ，因此该数据管道只支持Qwen2-VL，训练其他模型需要自定义这两个类

### 2. 准备模型
从hugging face中下载模型：  
```plaintext
huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct --local-dir your-path
```


### 3. 设定参数，运行sh脚本
根据系统配置调整参数，运行sft.sh脚本

### 致谢
以下开源项目对本工作提供了重要支持：
- [Transformers](https://github.com/huggingface/transformers): State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): Qwen2.5-VL is the multimodal large language model series developed by Qwen team, Alibaba Cloud.
- [LLaVA](https://github.com/haotian-liu/LLaVA): Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond.
