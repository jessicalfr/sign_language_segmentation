# Segmentação de Língua de Sinais com base em pose

O modelo utilizado nessa solução é baseado no modelo apresentado no artigo ["Real-Time Sign Language Detection using Human Pose Estimation"](https://slrtp.com/papers/full_papers/SLRTP.FP.04.017.paper.pdf).


## Ambiente de execução

Esse repositório contém um dockerfile com a descrição de um ambiente docker que faz a execução de todos os experimentos. Foi utilizada uma placa de vídeo NVIDIA GeForce RTX 3050.

## Pré-processamento dos dados

O pré-processamento organiza os dados para treino e validação no formato `tfrecord` que contém os seguintes campos:
- `fps` (`Int64List`): framerate do vídeo
- `pose_data` (`BytesList`): estimativa de pose, em um tensor do formato `(frames, 1, keypoints, 2)`
-  `pose_confidence` (`BytesList`): confiança da estimativa de pose, em um tensor do formato `(frames, 1, keypoints)`
- `is_signing` (`BytesList`): representação se a pessoa estava sinalizando (1) ou não (0) a cada frame

Os valores das coordenadas dos *keypoints* são normalizados em relação à resolução do vídeo.

Para o processamento do Public DGS Dataset:

`python3 ./preprocessing/build_tfrecord_train.py --skel <pasta-jsons-esqueleto> --type_skel DGS --annot <pasta-anotacoes-eaf> --fps 50 --output <pasta-output>`

Foram utilizados apenas as câmeras A e B do dataset, em que aparecem apenas uma pessoa por vez. Foram removidos do dataset os vídeos que não continham o tier correspondente no arquivo de anotação do ELAN (`Deutsche_Übersetzung_A` ou `Deutsche_Übersetzung_B` para as câmeras A e B, respectivamente).

O dado já processado pode ser baixado [nesse link](https://ufmgbr-my.sharepoint.com/:u:/g/personal/jessicalfr_ufmg_br/EYH2wSRK-SRFt0RWwLlWSAwB5IkzGeejgO2SU5ruBVRsEg?e=1GkHKF).


## Experimentos

O dataset pré-processado contém 570 sequências de esqueletos extraídos dos vídeos, que foram divididos em 342 sequêndias para treino (60%), 114 sequências para validação (20%) e 114 sequências para teste (20%). Cada sequência contém, em média 6,8 minutos de duração a 50 fps.

A proporção de frames com label 1 (a pessoa está sinalizando) no dataset é dada pela tabela. Podemos ver que o dataset é balanceado.

**Dataset** | **Y=1 no total de frames** | **Média de Y=1 por sequência** | **Desvio padrão de Y=1 por sequência** |
--------|:-----:|:-----:|:-----:|
Treino  | 0,503 | 0,535 | 0,267 |
Dev     | 0,516 | 0,564 | 0,246 |
Teste   | 0,501 | 0,549 | 0,259 |
**Total**   | **0,506** | **0,543** | **0,261** |

