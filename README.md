# Adversarial_attack_on_deception_detection (Pytorch)
This repo is for the final project of the course Security and Privacy of Machine Learning (SPML).

# Motivation
Recently, some papers have been exploring the use of faces and audios for deception detection. However, the adversarial attack on the face or audio features has not been studied. We design some models of the sever and attacker for verifying whether the adversarial attack exists in the deception detection task. 

# Model design
The server model is illustrated below. Where the left figure is a single_modal model (only images), and the right figure is a multi_modal model (both images and audios). Here the CNN backbone is resnet18 or resnext50, and the sequential processing module is GRU or Transformer.
<p align="center">
<img src="https://github.com/come880412/Adversarial_attack_on_deception_detection/blob/main/images/server_model.jpg" width=30% height=30%>
  <img src="https://github.com/come880412/Adversarial_attack_on_deception_detection/blob/main/images/sever_multi_modal_model.jpg" width=32% height=32%>
</p>
We also have the attacker model, which is illustrated below. Where the left figure is a video classifer, and the right figure is an audio classifier. Here the CNN backbone is Alexnet, VGG16, resnet18, resnext50, and InceptionNetv3.
<p align="center">
  <img src="https://github.com/come880412/Adversarial_attack_on_deception_detection/blob/main/images/Attacker_image_model.jpg" width=30% height=30%>
  <img src="https://github.com/come880412/Adversarial_attack_on_deception_detection/blob/main/images/Attacker_audio_model.jpg" width=20% height=20%>
</p>

# Attack algorithm
Here we apply three different attack algorithm to generate the adversarial examples.
- Fast Gradient Sign Method (FGSM) [1]
- Iterative FGSM (I-FGSM) [2]
- Momentum Iterative FGSM (MI-FGSM) [3]

# Dataset
- Real-life trial dataset: We use the publicly-available dataset [4] to conduct experiments. The dataset contains 121 videos, including 61 deceptive videos and 60 truth videos, collected from the court. 

# Experiments
The performance on the server model:
| | ResNet18_GRU | ResNest18_Transformer | ResNext50_GRU | ResNext50_Transformer |
|:----------:|:----------:|:----------|:----------|:----------|
| Video | 78.18% | 90.91% | 81.82% | 86.36% |
| Video + Audio | _**95.45%**_ | _**90.91%**_ | - | - |

The performance on the attacker model:
| | AlexNet | VGG16 | ResNet18 | ResNet50 | ResNext50 |
|:----------:|:----------:|:----------|:----------|:----------|:----------|
| Video | 72.73% | 89.10% | 84.55% | _**90.91%**_ | _**90.91%**_ |
| Audio | 72.73% | 72.73% | 81.82% | 90.91% | _**100%**_ |

Adversarial attack on the sever of Video model:
| | ResNet18_GRU | ResNest18_Transformer | ResNext50_GRU | ResNext50_Transformer |
|:----------:|:----------:|:----------|:----------|:----------|
| Standard Acc. | 78.18% | 90.91% | 81.82% | 86.36% |
| ResNet18 (fgsm) | 71.82% | 63.64% | 48.18% | 63.64% |
| Ensemble (fgsm) | 67.27% | 63.64% | _**46.36%**_ | 63.64% |
| ResNet18 (i-fgsm) | 68.18% | 81.82% | 71.82% | 72.73% |
| Ensemble (i-fgsm) | 74.55% | 63.64% | 73.64% | 51.82% |
| ResNet18 (mi-fgsm) | 70.91% | 55.45% | 64.55% | 60.91% |
| Ensemble (mi-fgsm) | _**66.36%**_ | _**45.45%**_ | 61.82% | _**49.09%**_ |

Adversarial attack on the server of multi_modal model:
| Video attack only | Standard Acc. | ResNet18 (fgsm) | Ensemble (fgsm) | ResNet18 (i-fgsm) | Ensemble (i-fgsm) | ResNet18 (mi-fgsm)| Ensemble (mi-fgsm) |
|:----------:|:----------:|:----------|:----------|:----------|:----------|:----------|:----------|
| ResNet18_GRU | 95.45% | 92.73% | 88.18% | 93.64% | 79.09% | 92.73% | _**70.00%**_ |
| ResNest18_Transformer | 90.91% | 87.27% | 80.00% | 90.00% | 70.91% | 82.73% | _**67.27%**_ |

| Audio attack only | Standard Acc. | ResNet18 (fgsm) | Ensemble (fgsm) | ResNet18 (i-fgsm) | Ensemble (i-fgsm) | ResNet18 (mi-fgsm)| Ensemble (mi-fgsm) |
|:----------:|:----------:|:----------|:----------|:----------|:----------|:----------|:----------|
| ResNet18_GRU | 95.45% | 94.55% | 95.45% | 91.82% | 91.82% | _**90.91%**_ | 92.73% |
| ResNest18_Transformer | 90.91% | 90.00% | 90.91% | 90.00% | 90.91% | _**90.00%**_ | 90.91% |

| multi_modal attack | Standard Acc. | ResNet18 (fgsm) | Ensemble (fgsm) | ResNet18 (i-fgsm) | Ensemble (i-fgsm) | ResNet18 (mi-fgsm)| Ensemble (mi-fgsm) |
|:----------:|:----------:|:----------|:----------|:----------|:----------|:----------|:----------|
| ResNet18_GRU | 95.45% | 90.00% | 82.73% | 87.27% | 73.64% | 82.73% | _**68.18%**_ |
| ResNest18_Transformer | 90.91% | 80.91% | 80.00% | 88.18% | 63.64% | 66.36% | _**55.45%**_ |

Adversarial training on ResNet18-fgsm adv. samples:
| | ResNet18_GRU | ResNest18_Transformer | ResNext50_GRU | ResNext50_Transformer |
|:----------:|:----------:|:----------|:----------|:----------|
| Standard Acc. | 89.09% | 88.18% | 90.91% | 87.27% |
| ResNet18 (fgsm) | _**90.00%**_ | _**89.09%**_ | _**90.91%**_ | _**81.82%**_ |
| Ensemble (fgsm) | 83.64% | 88.18% | 88.18% | 77.27% |

Adversarial training on Ensemble-fgsm adv. samples:
| | ResNet18_GRU | ResNest18_Transformer | ResNext50_GRU | ResNext50_Transformer |
|:----------:|:----------:|:----------|:----------|:----------|
| Standard Acc. | 81.82% | 79.09% | 79.09% | 81.82% |
| ResNet18 (fgsm) | 75.45% | 79.09% | 78.18% | 80.91% |
| Ensemble (fgsm) | _**78.18%**_ | _**85.45%**_ | _**79.09%**_ | _**81.82%**_ |

# Getting started
- Clone this repo to your local
``` bash
git clone https://github.com/come880412/Adversarial_attack_on_deception_detection.git
cd Adversarial_attack_on_deception_detection
```

### Download the dataset
You should first download the Real-Life trial dataset by sending the request to the author of [4]. After that, put the dataset to the current path.

### Training
We have server and attacker models, you can use the following script to train the models:
```bash
$ cd ./server 
$ python train.py --Sequential_processing GRU --backbone resnet18 --data_path path/to/dataset --batch_size 4

$ cd ../attacker
$ python main.py --input_type image --action pretrain_image --backbone resnet18 --data_path path/to/dataset --image_path path/to/dataset/Real_life --csv_path path/to/dataset/Real_life.csv --save_adv_path path/to/dataset/Real_life_adv
```
- If you have any implementation problem, feel free to E-mail me! come880412@gmail.com

# References
[1] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014. \
[2] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017. \
[3] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial attacks with momentum. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9185–9193, 2018. \
[4] Pérez-Rosas, V., Abouelenien, M., Mihalcea, R., & Burzo, M. Deception detection using real-life trial data. International Conference on Multimodal Interaction, 2015
