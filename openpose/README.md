# Pré-Processamento: Extração de Esqueletos

A extração de esqueletos dos vídeos é feita com o [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). O dockerfile foi construído baseado no repositório [STomoya/openpose-docker](https://github.com/STomoya/openpose-docker).

## Container Docker

O container pode ser criado com o comando `docker build -t openpose .`.

Para iniciar o container basta executar `docker run -it --runtime=nvidia --gpus=all -v <folder>:/<folder> openpose`.

## Execução para um vídeo

Para extrair os 137 keypoints (body, face, hands) em um vídeo basta executar `bin/openpose.bin --video <video-path.mp4> --face --hand --write_json <output-folder> --display 0 --render_pose 0 --model_folder /usr/local/openpose/models/`. Cada frame do vídeo vai gerar um arquivo JSON contendo as coordenadas dos *keypoints* e o score de confiança da estimativa.

## Execução para vários vídeos

Para extrair os esqueletos de vários vídeos em sequência, basta executar `python3 extract_skel.py --input_folder <folder-with-videos> --output_folder <folder-to-save-keypoints>`.