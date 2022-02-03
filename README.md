## In this repository, I will be showing and end-to-end (less data collection) object detection project using detectron2. First, we will be preparing the dataset and training the model. Once the model is trained, we will serialize the model so for performance also removing python dependencies so it can be deployed on other runtimes like C++. Then we will be creating our custom detectron2 handlers (basically, preprocessing, inference, postprocessing) as we want the server to be be handling this so the client can be more lightweight. Lastly we will be creating a simple UI and then automate the deployment using docker-compose

#### Overview of the entire process
<img src="/assets/overview.svg">


## Model preparation
#### Kaggle notebook links
As the dataset is store in kaggle, I have provided the notebook links so anyone can run the notebooks. 

1. Dataprep notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-dataprep) 
2. Training notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-training)
3. Submission notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-submission)
4. Torchscript notebook (https://www.kaggle.com/georgeteo89/satorius-segmentation-torchscript)


## Deployment on TorchServe
In order to deploy on Torchserve, I have done the following 
1. Creating a custom handler script 
2. Packaging the custom handler and the scripted model into a MAR file that is required for torchserve
3. Created a new torchserve docker image with detectron2 installed and the MAR file added so the image is production ready out of the box

## UI code
I have created a simple StreamLit UI for testing out if the model is serving correctly. The UI have also been dockerized.


## Test out for yourself!
I have created a docker-compose file so ease of deploying