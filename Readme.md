# [ICCV 2025] HOLa: 基于低秩分解 VLM 特征适应的零样本 HOI 检测

## 论文链接
[arXiv](https://arxiv.org/abs/2507.15542)
[项目页面](https://chelsielei.github.io/HOLa_Proj/)

## 数据集
请遵循 [UPT](https://github.com/fredzzhang/upt) 的流程。
下载的文件应按以下方式放置。否则，请将默认路径替换为您自定义的位置。
```
|- HOLa
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

## 依赖
1.  遵循 [UPT](https://github.com/fredzzhang/upt) 中的环境设置。
2.  遵循 [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main) 中的环境设置。
**提醒**：
如果您已经在 Python 环境中安装了 `clip` 软件包（例如，通过 `pip install clip`），请确保使用我们 EZ-HOI 仓库中提供的本地 CLIP 目录。为此，请设置 `PYTHONPATH` 以包含本地 CLIP 路径，使其优先于已安装的软件包。
```
export PYTHONPATH=$PYTHONPATH:"your_path/HOLa/CLIP"
```
这样您就可以使用本地 `clip` **而无需卸载您 Python 环境中的 `clip`**。

3.  按照 [这里](https://github.com/ChelsieLei/EZ-HOI/issues/2) 提到的修改已安装的 [pocket](https://github.com/fredzzhang/pocket) 库。
## 脚本
### 在 HICO-DET 上训练/测试：
使用 **vit-B** 图像骨干网络：
```
bash scripts/hico_vitB.sh
```
使用 **vit-L** 图像骨干网络
```
bash scripts/hico_vitL.sh
```
### 在 V-COCO 上训练/测试：
使用 **vit-L** 图像骨干网络：
```
bash scripts/vcoco.sh
```

## 模型库
| 数据集 | 设置 | 骨干网络 | mAP | Unseen | Seen |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HICO-DET | UV | ResNet-50+ViT-B | 34.09 | 27.91 | 35.09 |
| HICO-DET | RF | ResNet-50+ViT-B | 34.19 | 30.61 | 35.08 |
| HICO-DET | NF | ResNet-50+ViT-B | 32.36 | 35.25 | 31.64 |
| HICO-DET | UO | ResNet-50+ViT-B | 33.59 | 36.45 | 33.02 |

| 数据集 | 设置 | 骨干网络 | mAP | Rare | Non-rare |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HICO-DET | default | ResNet-50+ViT-B | 35.41 | 34.35 | 35.73 |
| HICO-DET | default | ResNet-50+ViT-L | 39.05 | 38.66 | 39.17 |

您可以使用以下 Google Drive 链接下载我们预训练的模型检查点：
```
https://drive.google.com/drive/folders/1kH-yOi-YqdB35rSgKoRkmg_pGbyFEkUX?usp=sharing
```
您也可以使用以下夸克链接下载我们预训练的模型检查点：
```
链接: https://pan.quark.cn/s/c3f30b122ed2
提取码: yawa
```

## 引用
如果您觉得我们的论文和/或代码有所帮助，请考虑引用：
```
@inproceedings{
lei2025hola,
title={HOLa: Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation},
author={Lei, Qinqian and Wang, Bo and Robby T., Tan},
booktitle={In Proceedings of the IEEE/CVF international conference on computer vision},
year={2025}
}
```
## 致谢
我们衷心感谢 [UPT](https://github.com/fredzzhang/upt) 和 [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main) 的作者们开源了他们的代码。
-----