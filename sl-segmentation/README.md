# Segmentação de Língua de Sinais com base em pose

O modelo utilizado nessa solução é baseado no modelo apresentado no artigo ["Real-Time Sign Language Detection using Human Pose Estimation"](https://slrtp.com/papers/full_papers/SLRTP.FP.04.017.paper.pdf).


## Ambiente de execução

Esse repositório contém um dockerfile com a descrição de um ambiente docker que faz a execução de todos os experimentos. As execuções foram feitas utilizando uma placa de vídeo NVIDIA GeForce RTX 3050.

## Pré-processamento dos dados

O pré-processamento organiza os dados para treino e validação no formato `tfrecord` que contém os seguintes campos:
- `fps` (`Int64List`): framerate do vídeo
- `pose_data` (`BytesList`): estimativa de pose, em um tensor do formato `(frames, 1, keypoints, 2)`
-  `pose_confidence` (`BytesList`): confiança da estimativa de pose, em um tensor do formato `(frames, 1, keypoints)`
- `is_signing` (`BytesList`): representação se a pessoa estava sinalizando (1) ou não (0) a cada frame

Os valores das coordenadas dos *keypoints* são normalizados em relação à resolução do vídeo.

Para o processamento do Public DGS Dataset:

`python3 ./preprocessing/build_tfrecord_train.py --skel <pasta-jsons-esqueleto> --type_skel DGS --annot <pasta-anotacoes-eaf> --fps 50 --output <pasta-output>`

O dado já processado pode ser baixado [nesse link](https://ufmgbr-my.sharepoint.com/:u:/g/personal/jessicalfr_ufmg_br/EYH2wSRK-SRFt0RWwLlWSAwB5IkzGeejgO2SU5ruBVRsEg?e=1GkHKF).