# Sign Language Segmentation

Sistema de segmentação de vídeos de língua de sinais em clips que contenham trechos equivalentes a sentenças na língua escrita.

## Organização do repositório

- `sl-segmentation`: Implementação do modelo e dos experimentos.
- `openpose`: Implementação de extração de esqueletos utilizando OpenPose.
- `utils`: Utilidades gerais do processamento dos dados.


## Baseline

**Real-Time Sign Language Detection using Human Pose Estimation ([Código](https://github.com/google-research/google-research/tree/master/sign_language_detection))**

Uma implementação em Tensorflow do modelo apresentado em ["Real-Time Sign Language Detection using Human Pose Estimation"](https://slrtp.com/papers/full_papers/SLRTP.FP.04.017.paper.pdf), publicado no SLRTP 2020.


## Datasets utilizados
**Public DGS Corpus**

O [Public DGS Corpus](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html) contém 50 horas de vídeos do projeto DGS-Korpus de diferentes tópicos. Contém vídeos de pessoas sinalizando em Língua de Sinais Alemã, os esqueletos extraídos com OpenPose, além de anotações das sentenças em alemão e suas traduções em inglês.

**MEDIAPI-SKEL**

O [MEDIAPI-SKEL](https://www.ortolang.fr/market/corpora/mediapi-skel/) é um dataset de Língua de Sinais Francesa de 27 horas de vídeos. Ele contém os esqueletos extraídos com OpenPose dos vídeos originais, além da anotação das sentenças em francês.

## Ferramentas utilizadas
**OpenPose ([Código](https://github.com/CMU-Perceptual-Computing-Lab/openpose))**

Sistema de detecção de *keypoints* do tronco, mãos, rosto e pés em imagens.