# Importing necessary dependencies
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce

# Creating a Spark Session 
spark = SparkSession.builder.master("local[*]").getOrCreate()

# Checking for command line arguments passed
# If a file is passed via command line, use that for prediction using model
# Else throw an error
try:
    filepn = "/job/" + str(sys.argv[1])
    data_test = spark.read.option("delimiter", ";").csv(filepn, header=True, inferSchema=True)
    print("File input :", str(sys.argv[1]))

except:
    exit()

# To clean out CSV headers if quotes are present
old_column_name = data_test.schema.names
print(data_test.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"', ''))

data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]),
                   range(len(clean_column_name)), data_test)
print(data_test.schema)

# Create a PipelineModel object to load saved model parameters from Train
try:
    PipeModel = PipelineModel.load("/job/Modelfile")
except:
    exit()

# Generate predictions for Input dataset file
try:
    test_prediction = PipeModel.transform(data_test)
except:
    print("---")

# Save the resulting predictions with the original dataset to a CSV File
test_prediction.drop("feature", "Scaled_feature", "rawPrediction", "probability").write.mode("overwrite").option(
    "header", "true").csv("/job/resultdata.csv")
# test_prediction.select("quality", "prediction").write.mode("overwrite").option("header", "true").csv("/job/resultdata")

# Creating an evaluator classification object to generate metrics for predictions
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

# Calculating the F1 score/Accuracy of the model with Test dataset
test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

print("[Test] F1 score =", test_F1score)
print("[Test] Accuracy =", test_accuracy)

# Save the results onto a Text File called results.txt
fp = open("/job/results.txt", "w")
fp.write("[Test] F1 score =  %s\n" % test_F1score)
fp.write("[Test] Accuracy =  %s\n" % test_accuracy)

# Closing the file
fp.close()
