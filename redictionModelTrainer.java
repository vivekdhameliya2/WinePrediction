/**
 * This is Programming assignment 2 of CS 643 - Cloud computing.
 * A wine quality prediction ML model in Spark over AWS.
 * For this project, an AWS EMR cluster is used with one master EC2 and three Slave EC2 VMs.
 * 
 * @author Milankumar Patel (mkp6) 
 **/

package winequalityprediction;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.HashMap;

public class PredictionModelTrainer {
    
    /*
     * getPerformance() - Evaluates the output for the dataset and finds the Mean absolute error in float
     * @param testData 
     */
    public void getPerformance(Dataset<Row> dataset){
        RegressionEvaluator regressionEvaluator = new RegressionEvaluator().setLabelCol("quality").setPredictionCol("prediction").setMetricName("mae");
        
        // Evaluates the output for the dataset
        double meanAbsoluteError = regressionEvaluator.evaluate(dataset);
        
        System.out.println("----------------------------------------------------");
        System.out.println("------ Mean absolute error = " + meanAbsoluteError + "------");
        System.out.println("----------------------------------------------------");
    }

    /*
     * loadDataset() - this function loads the data from csv and removes all duplicate values from it
     * @param spark
     * @param csvFile
     * @return cleanDataSet cleaned Dataset of CSV file data
     */
    public Dataset<Row> loadDataset(SparkSession spark, String csvFile){

        HashMap<String, String> fileOptions = new HashMap<>();
        fileOptions.put("delimiter", ";");
        fileOptions.put("inferSchema", "true");
        fileOptions.put("header", "true");

        /*
         * load the csv into memory for the current row
         */
        Dataset<Row> rowDataSet = spark.read().options(fileOptions).csv(csvFile);

        /*
         * Returns a new Dataset that contains only the unique rows from this Dataset.
         */
        Dataset<Row> cleanDataSet = rowDataSet.dropDuplicates();
        return cleanDataSet;
    }
    
    
    /*
     * transformDataset() this function will transform the cleaned dataset into Vector
     * @param cleanedDataSet cleaned Dataset of CSV file data
     * @return transformedDataSet 
     */
    public Dataset<Row> transformDataset(Dataset<Row> cleanedDataSet){
        /*
         * Selects a set of column
         */
        Dataset<Row> featureColumns = cleanedDataSet
        		.select("fixed acidity", "volatile acidity", "citric acid",
        				"residual sugar", "chlorides", "free sulfur dioxide",
        				"total sulfur dioxide", "density", "pH", "sulphates", "alcohol");

        /*
         * vectorAssembler transformer that merges multiple columns into a vector column.
         */
        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(featureColumns.columns()).setOutputCol("features");

        /*
         * Chaining custom transformations for datasets
         * transform dataset in vector type
         */
        Dataset<Row> transformedDataSet = vectorAssembler.transform(cleanedDataSet).select("features", "quality").cache();

        return transformedDataSet;
    }
    
    
    
    /*
     * Main function for PredictionModelTrainer
     */
    public static void main(String[] args) {
        
        /*
         * Most logging operations, except configuration, are done through this Logger Interface
         */
        Logger.getLogger("org").setLevel(Level.ERROR);
        
        /*
         * Check for the required arguments for the Model
         */
        if (args.length < 3) {
            System.err.println("This ML training model requires three arguments: training set, validation set, prediction model path");
            System.exit(1);
        }
        
        /*
         * Initialization of objects and get the parameters
         */
        final String training_dataset = args[0];
        final String validation_dataset = args[1];
        final String predictionModelPath = args[2];
 
        PredictionModelTrainer predictionModelTrainer = new PredictionModelTrainer();
        
        /*
         * Sets a name for the application, if there is no existing one, creates a new one
         */
        SparkSession sparkSession = new SparkSession.Builder().appName("Wine quality model").getOrCreate();
        
        /*
         * loads the data from csv and removes all duplicate values from it
         */
        Dataset<Row> trainingLoadDataSet = predictionModelTrainer.loadDataset(sparkSession, training_dataset);
        
        /*
         * this function will transform the cleaned dataset into Vector
         */
        Dataset<Row> trainingFinalDataSet = predictionModelTrainer.transformDataset(trainingLoadDataSet);
        
        /*
         * Performs a linear regression on the data points
         */
        LinearRegression linearRegression = new LinearRegression().setMaxIter(20).setRegParam(0).setFeaturesCol("features").setLabelCol("quality");

        /*
         * Configure an ML pipeline, which consists of a stage for the training model
         */
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{linearRegression});
        
        /*
         * Fit the pipeline to training csv.
         */
        PipelineModel model = pipeline.fit(trainingFinalDataSet);

        /*
         * Loads and transforms the validation data set
         */
        Dataset<Row> loadValidationsData = predictionModelTrainer.loadDataset(sparkSession, validation_dataset);
        Dataset<Row> finalValidationData = predictionModelTrainer.transformDataset(loadValidationsData);

        /*
         * Make predictions on test csv.
         */
        Dataset<Row> predictions = model.transform(finalValidationData);
        predictions.show();

        /*
         * Evaluate Model performance
         */
        predictionModelTrainer.getPerformance(predictions);

        /*
         * Save the developed model
         */
        try{
            model.write().overwrite().save(predictionModelPath);
        } catch (IOException e){
            System.out.println("Something went wrong when writing the model to the disk. " + e.getMessage());
        }
    }
}
