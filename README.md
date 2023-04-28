# Sign Language Segmentation

Sistema de segmentação de vídeos de língua de sinais em clips que contenham trechos equivalentes a sentenças na língua escrita.

## Para executar inferência

*Em andamento*

## Para reproduzir o treino

*Em andamento*


## Baseline: Real-Time Sign Language Detection using Human Pose Estimation
[Artigo](https://slrtp.com/papers/full_papers/SLRTP.FP.04.017.paper.pdf) | [Código](https://github.com/google-research/google-research/tree/master/sign_language_detection)

Uma imprementação em Tensorflow do modelo apresentado em "Real-Time Sign Language Detection using Human Pose Estimation", publicado no SLRTP 2020.

> Amit Moryossef, Ioannis Tsochantaridis, Roee Aharoni, Sarah Ebling, & S. Narayanan (2020). Real-Time Sign Language Detection using Human Pose Estimation. In Sign Language Recognition, Translation & Production (SLRTP 2020).


## Datasets
### Public DGS Corpus

O [Public DGS Corpus](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html) contém 50 horas de vídeos do projeto DGS-Korpus de diferentes tópicos. Contém vídeos de pessoas sinalizando em Língua de Sinais Alemã, os esqueletos extraídos com OpenPose, além de anotações das sentenças em alemão e suas traduções em inglês.

> Konrad, R., Hanke, T., Langer, G., Blanck, D., Bleicken, J., Hofmann, I., Jeziorski, O., König, L., König, S., Nishio, R., Regen, A., Salden, U., Wagner, S., Worseck, S., Böse, O., Jahn, E., Schulder, M. 2020. MEINE DGS – annotiert. Öffentliches Korpus der Deutschen Gebärdensprache, 3. Release / MY DGS – annotated. Public Corpus of German Sign Language, 3rd release [Dataset]. Universität Hamburg. https://doi.org/10.25592/dgs.corpus-3.0

### MEDIAPI-SKEL

O [MEDIAPI-SKEL] (https://www.ortolang.fr/market/corpora/mediapi-skel/) é um dataset de Língua de Sinais Francesa de 27 horas de vídeos. Ele contém os esqueletos extraídos com OpenPose dos vídeos originais, além da anotação das sentenças em francês.

> Bull, H., Braffort, A. and Gouiffès, M. (2020). MEDIAPI-SKEL - A 2D-Skeleton Video Database of French Sign Language With Aligned French Subtitles. In Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC'2020), Marseille, France, May.

## Ferramentas
### OpenPose
[Código](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

Sistema de detecção de *keypoints* do corpo humano, mãos, rosto e pés em imagens.

> Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, & Y. A. Sheikh (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. IEEE Transactions on Pattern Analysis and Machine Intelligence.

> Tomas Simon, Hanbyul Joo, Iain Matthews, & Yaser Sheikh (2017). Hand Keypoint Detection in Single Images using Multiview Bootstrapping. In CVPR.

> Zhe Cao, Tomas Simon, Shih-En Wei, & Yaser Sheikh (2017). Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. In CVPR.

> Shih-En Wei, Varun Ramakrishna, Takeo Kanade, & Yaser Sheikh (2016). Convolutional pose machines. In CVPR.