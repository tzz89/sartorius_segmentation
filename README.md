## Objection detection using detectron2
In this repository, I will be showing an end-to-end (less data collection) object detection project using detectron2. First, we will be preparing the dataset and training the model. Once the model is trained, we will serialize the model for performance and to remove python dependencies so it can be deployed on other runtimes like C++. Then we will be creating our custom detectron2 handlers that is required for torchserve (basically, how we want to do preprocessing, forward passing , postprocessing). If we perform these operations on the server side, the client side will be free from installing a mountain of dependencies. Lastly we will be creating a simple UI and then link the containers up using docker-compose. Docker compose will handle the networking for us.

#### Overview of the entire process
<img src="/assets/overview.svg">


## Model preparation
#### Kaggle notebook links
As the dataset is stored in kaggle, I have provided the notebook links so anyone can run the notebooks without downloading the dataset

1. Dataprep notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-dataprep) 
2. Training notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-training)
3. Submission notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-submission)
4. Torchscript notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-torchscript)


## Deployment on TorchServe
In order to deploy on Torchserve, I have done the following 
1. Creating a custom handler script 
2. Packaging the custom handler and the scripted model into a MAR file that is required for torchserve
3. Created a new torchserve docker image with detectron2 installed and the MAR file added so the image is ready for use out of the box

dockerhub image: https://hub.docker.com/repository/docker/teozz89/sartorius

## UI code
I have created a simple StreamLit UI for testing out if the model is serving correctly. The UI have also been dockerized.

dockerhub image: https://hub.docker.com/repository/docker/teozz89/sartorius_ui


## Test out for yourself!
I have created a docker-compose file so ease of deploying
Steps:
1. git clone this project
2. cd to the project folder 
3. open up a cmd prompt and run "docker-compose -f docker-compose.yaml up"
<img src="/assets/docker_desktop.JPG">
4. open a browser and go to localhost:8051
5. drop in some sample images from the sample_images folder


<img src="/assets/drag_drop.JPG">

<img src="/assets/uploaded_image.JPG">

<img src="/assets/predictions.JPG">
