# Scene Change Detection Server

This repository includes a simple Flask server detecting scene changes from given old and new images.

## Scene Change Detection Model

Model was trained based on [DR-TANet](https://github.com/Herrccc/DR-TANet) repository. "VL-CMU-CD" dataset
from [Street-View Change Detection with Deconvolutional Networks](http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf)
was used during the training.

You can download the pre-trained model
from [[googledrive]](https://drive.google.com/file/d/1IvGD0gbBxcM72hmiFd_6yYXZ1UKEWDHY).

## Build Docker image from Dockerfile

```
docker image build -t scd-server .
```

## Run a Docker container

```
docker run -p 5000:5000 -d --name scd-server  scd-server
```

## API usage

```
Method: POST
Endpoint: /ai/detect
Request Payload:
    {
        old_im: <base64 formatted string>
        new_im: <base64 formatted string>
    }
Response Payload:
    {
        data: {
            result: <base64 formatted string>
        },
        info: {
            code: <response code>,
            message: <response message>
        }
    }
```