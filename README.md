# Wine Quality Prediction Using Pyspark and Amazon Web Services (AWS)

This guide explains the procedure to use AWS services to train an ML (Machine Learning) model on multiple parallel EC2 (Elastic Compute Cloud) instances. The ML program is written in Python using Apache Spark MLlib libraries. The training and prediction programs are configured to run inside a container.

## Table of Contents
1. [Setting up EC2 Cluster on AWS](#setting-up-ec2-cluster-on-aws)
2. [Setting up Task Definitions and Tasks](#setting-up-task-definitions-and-tasks)
3. [Running the Prediction Application on AWS with Docker](#running-the-prediction-application-on-aws-with-docker)
4. [Running the Prediction Application without Docker](#running-the-prediction-application-without-docker)
5. [Using WinSCP to Transfer Data](#using-winscp-to-transfer-data)

   
## Links
- **GitHub Repository**: [WinePrediction](https://github.com/vivekdhameliya2/WinePrediction)
- **Docker Container link**: [winequality](https://hub.docker.com/repository/docker/vivekdhameliya/winequality)

## Setting up EC2 Cluster on AWS
To run the ML container application for training on multiple parallel EC2 instances, a cluster needs to be set up. Follow these steps to create a cluster with 4 instances:

- In AWS Management Console, search for **Elastic Container Service (ECS)** and click on it.
- In ECS Console, select **Cluster** and click on *"Create Cluster"*.
- Choose the *EC2 Linux + Networking* cluster template.
- Configure the cluster parameters and click on *"Create"*.

Once the cluster is created, 4 EC2 instances will be registered to the cluster.

## Setting up Task Definitions and Tasks
To run the ML container application, you need to create a "Task Definition" that describes information regarding the containers to use, bind mounts, volumes, etc. Follow these steps:

- In **ECS console**, select *"Task Definitions"*.
- Click on *"Create New Task Definition"*.
- Choose **Select launch type compatibility** as "EC2".
- Configure the task and container definitions, specifying volumes and container paths.

We have successfully created our *"Task Definition"*. Now we need to create a Task that will initiate our Docker container application on the EC2 instances.

## Running the Prediction Application on AWS with Docker
The ML container for prediction uses two files as input: *"Modelfile"* and *"TestDataset.csv"*. Follow these steps to run the prediction application:

### Launching an instance on AWS
- Launch an Ubuntu instance on AWS.

### Installing Docker and Downloading the Prediction Container
```bash
sudo yum install docker -y && sudo systemctl start docker
docker push vivekdhameliya/winequality:tagname

Running the Docker Container

sudo docker run -v /home/ec2-user/:/job vivekdhameliya/winequality:latest TestDataset.csv


### running-the-prediction-application-without-docker
To run the prediction application without Docker, you need to install required packages, including Pyspark, Java JDK, numpy, and Apache Spark.

Java Installation
Download Java JDK from Oracle.
Install Java JDK and set up environment variables.
Installing Apache Spark
Download Apache Spark from Spark Website.
Extract the tar file and set up environment variables.


###Running Prediction Application
python3 wine_test_nodocker.py TestDataset.csv

Using WinSCP to Transfer Data
To transfer data between the instance and the local PC, use WinSCP. Configure and connect to the instance using the ppk key and transfer files as needed.

