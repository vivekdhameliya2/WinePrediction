FROM bde2020/spark-maven-template:3.3.0-hadoop3.3

MAINTAINER Milankumar Patel <mkp6@njit.edu>

ENV SPARK_VERSION=3.3.0

ENV HADOOP_VERSION=3

ENV SPARK_APPLICATION_JAR_NAME winequalityprediction.jar

ENV SPARK_APPLICATION_MAIN_CLASS PredictionModelTrainer.TestPrediction

ENV SPARK_APPLICATION_ARGS "file:///opt/workspace/Test-File.csv file:///opt/workspace/model"

VOLUME /opt/workspace
