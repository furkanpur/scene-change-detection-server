FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# Timezone
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# system packages
RUN apt-get update
RUN apt-get install git curl python3-dev python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# pip upgrade
RUN pip3 install torch torchvision torchaudio
RUN pip3 install gdown

WORKDIR /opt

RUN git clone https://github.com/furkanpur/scene-change-detection-server.git

WORKDIR /opt/scene-change-detection-server

RUN pip3 install -r requirements.txt

RUN mkdir -p resource
RUN gdown https://drive.google.com/uc?id=1IvGD0gbBxcM72hmiFd_6yYXZ1UKEWDHY
RUN mv model_00186600.pth resource

ENTRYPOINT ["python3"]

CMD ["server.py"]