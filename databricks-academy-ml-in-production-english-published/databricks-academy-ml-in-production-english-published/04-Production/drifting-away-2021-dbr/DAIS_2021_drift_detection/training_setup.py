# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Setup Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Use [notebook scoped libraries]() to install required dependency versions

# COMMAND ----------

# %pip install mlflow scipy==1.6.3 seaborn==0.11.1

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

from delta.tables import DeltaTable
import tempfile
import os
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Notebook Configs

# COMMAND ----------

# Get Databricks workspace username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
print(username)

# COMMAND ----------

# Set variables to use for reading/writing tmp artifacts and datasets
project_home_dir = f"/Users/{username}"
project_local_tmp_dir = "/dbfs" + project_home_dir + "tmp/"
data_project_dir = f"{project_home_dir}data/"

# Remove Data Project directory if exists - want to ensure there are no existing versions of Delta tables there
dbutils.fs.rm(data_project_dir, True)
dbutils.fs.mkdirs(data_project_dir)

# COMMAND ----------

# Set MLflow experiment path - require that we have created the folder DAIS_2021 in our user workspace 
workspace_project_home = f"/Users/{username}"

experiment_path = workspace_project_home + "/airbnb_hawaii"
if mlflow.get_experiment_by_name(experiment_path) is None:
    mlflow.create_experiment(experiment_path)

# Get the unique experiment ID from the provided
experiment_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

# Define model name for MLflow Registry
registry_model_name = "airbnb_hawaii"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Configs

# COMMAND ----------

# Path to Delta table of cleaned Airbnb Hawaii dataset
# This dataset was downloaded from http://insideairbnb.com/get-the-data.html 
# The cleaned (Delta) dataset can be found in the Google drive associated with bit.ly/dais_2021_drifting_away
# You can use the Databricks CLI https://docs.databricks.com/dev-tools/cli/index.html to upload this directory
# Download locally from the Google Drive, set up the CLI and use `dbfs cp -r airbnb-hawaii.delta dbfs:/path/to/dir/airbnb-hawaii.delta`
raw_delta_path = "dbfs:/dais-2021/airbnb-hawaii.delta"

# COMMAND ----------

raw_delta_path = "dbfs:/FileStore/tables/airbnb-hawaii.delta"

# COMMAND ----------

# dbutils.fs.rm("dbfs:/FileStore/tables/", recurse=True)
dbutils.fs.ls("dbfs:/FileStore/tables/airbnb-hawaii.delta")

# COMMAND ----------



# COMMAND ----------

# Paths to write/read data from
month_0_delta_path = data_project_dir + "month_0_delta"

# Two separate Data paths - one with error data/one without
month_1_error_delta_path = data_project_dir + "month_1_error_delta"
month_1_fixed_delta_path = data_project_dir + "month_1_fixed_delta"

month_2_delta_path = data_project_dir + "month_2_delta"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Dataset Creation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We will be creating synthetic errors in our datasets, simulating a case where we have 3 months of consecutive data, the first month being the dataset we use to train and "deploy" our first model, followed by 2 consecutive months where we have simulated different forms of drift. 

# COMMAND ----------

# Load full dataset and subset to specified columns
airbnb_df = spark.read.format("delta").load(raw_delta_path)

target_col = "price"
num_cols = ["accommodates",
            "bedrooms",
            "beds",
            "minimum_nights",
            "number_of_reviews",
            "number_of_reviews_ltm",
            "review_scores_rating"]
cat_cols = ["host_is_superhost",
            "neighbourhood_cleansed",
            "property_type",
            "room_type"]

cols_to_keep = [target_col] + num_cols + cat_cols
airbnb_df = airbnb_df.select(cols_to_keep)

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC Creating the data for each scenario

# COMMAND ----------

# The suffix of the variables used will correspond to the month they are intended to be used for
df_0, df_1, df_2 = airbnb_df.randomSplit(weights=[1.0, 1.0, 1.0], seed=42)

df_0.write.format("delta").save(month_0_delta_path)
df_1.write.format("delta").save(month_1_fixed_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 1 Creation
# MAGIC * Certain neighbourhoods are missing their `neighbourhood_cleansed` entries
# MAGIC * The upstream data generation procedure for `review_scores_rating` has resulted in the previously 0-100 rating system being altered to a 0-5 star system

# COMMAND ----------

# Create a DataFrame which takes the clean data and introduces simulated errors into the dataset
df_1_err = (df_1
             .withColumn("neighbourhood_cleansed",                                 # Simulate some neighbourhood entires as being cleansed incorrectly  
                         F.when((F.col("neighbourhood_cleansed") == "Primary Urban Center") |  
                                (F.col("neighbourhood_cleansed") == "Kihei-Makena") | 
                                (F.col("neighbourhood_cleansed") == "Lahaina") | 
                                (F.col("neighbourhood_cleansed") == "North Kona"), F.lit(None)).otherwise(F.col("neighbourhood_cleansed")))
            .fillna(0, subset=["review_scores_rating"])                            # Fill missing ratings with 0
            .withColumn("review_scores_rating", F.col("review_scores_rating")/20)  # Scale ratings to be between 0 and 5
            )

df_1_err.write.format("delta").save(month_1_error_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2:
# MAGIC * The new month of data contains listing entries recorded during peak vacation season. As a result, the price for every listing has been increased. 

# COMMAND ----------

df_2_err = df_2.withColumn("price", F.col("price") + (2*F.col("price")*F.rand(seed=42)))

df_2_err.write.format("delta").save(month_2_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Utility Functions

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### MLflow Setup

# COMMAND ----------

"""
MLflow Registry Clean Up 
"""

def cleanup_registered_model(registry_model_name):
  """
  Utilty function to delete a registered model in MLflow model registry.
  To delete a model in the model registry all model versions must first be archived.
  This function thus first archives all versions of a model in the registry prior to
  deleting the model
  
  :param registry_model_name: (str) Name of model in MLflow model registry
  """
  client = MlflowClient()

  filter_string = f'name="{registry_model_name}"'

  model_versions = client.search_model_versions(filter_string=filter_string)
  
  if len(model_versions) > 0:
    print(f"Deleting following registered model: {registry_model_name}")
    
    # Move any versions of the model to Archived
    for model_version in model_versions:
      try:
        model_version = client.transition_model_version_stage(name=model_version.name,
                                                              version=model_version.version,
                                                              stage="Archived")
      except mlflow.exceptions.RestException:
        pass

    client.delete_registered_model(registry_model_name)
    
  else:
    print("No registered models to delete")    

# COMMAND ----------

# Remove this registered model if it already exists
cleanup_registered_model(registry_model_name)

# COMMAND ----------

"""
MLflow Tracking Utility Methods
"""

def get_delta_version(delta_path):
  """
  Function to get the most recent version of a Delta table give the path to the Delta table
  
  :param delta_path: (str) path to Delta table
  :return: Delta version (int)
  """
  # DeltaTable is the main class for programmatically interacting with Delta tables
  delta_table = DeltaTable.forPath(spark, delta_path)
  # Get the information of the latest commits on this table as a Spark DataFrame. 
  # The information is in reverse chronological order.
  delta_table_history = delta_table.history() 
  
  # Retrieve the lastest Delta version - this is the version loaded when reading from delta_path
  delta_version = delta_table_history.first()["version"]
  
  return delta_version


def create_summary_stats_pdf(pdf):
  """
  Create a pandas DataFrame of summary statistics for a provided pandas DataFrame.
  Involved calling .describe on pandas DataFrame provided and additionally add
  median values and a count of null values for each column.
  
  :param pdf: pandas DataFrame
  :return: pandas DataFrame of sumary statistics for each column
  """
  stats_pdf = pdf.describe(include="all")

  # Add median values row
  median_vals = pdf.median()
  stats_pdf.loc["median"] = median_vals

  # Add null values row
  null_count = pdf.isna().sum()
  stats_pdf.loc["null_count"] = null_count

  return stats_pdf


def log_summary_stats_pdf_as_csv(pdf):
  """
  Log summary statistics pandas DataFrame as a csv file to MLflow as an artifact
  """
  temp = tempfile.NamedTemporaryFile(prefix="summary_stats_", suffix=".csv")
  temp_name = temp.name
  try:
    pdf.to_csv(temp_name)
    mlflow.log_artifact(temp_name, "summary_stats.csv")
  finally:
    temp.close() # Delete the temp file
    
    
def load_summary_stats_pdf_from_run(run, local_tmp_dir):
  """
  Given an MLflow run, download the summary stats csv artifact to a local_tmp_dir and load the
  csv into a pandas DataFrame
  
  :param run: mlflow.entities.run.Run
  :param local_tmp_dir: (str) path to a local filesystem tmp directory
  :return pandas DataFrame containing statistics computed during training
  """
  # Use MLflow clitent to download the csv file logged in the artifacts of a run to a local tmp path
  client = MlflowClient()
  if not os.path.exists(local_tmp_dir):
      os.mkdir(local_tmp_dir)
  local_path = client.download_artifacts(run.info.run_id, "summary_stats.csv", local_tmp_dir)
  print(f"Summary stats artifact downloaded in: {local_path}")
  
  # Load the csv into a pandas DataFrame
  summary_stats_path = local_path + "/" + os.listdir(local_path)[0]
  summary_stats_pdf = pd.read_csv(summary_stats_path, index_col="Unnamed: 0")
  
  return summary_stats_pdf 


def load_delta_table_from_run(run):
  """
  Given an MLflow run, load the Delta table which was used for that run,
  using the path and version tracked at tracking time.
  Note that by default Delta tables only retain a commit history for 30 days, meaning
  that previous versions older than 30 days will be deleted by default. This property can
  be updated using the Delta table property delta.logRetentionDuration.
  For more information, see https://docs.databricks.com/delta/delta-batch.html#data-retention
  
  :param run: mlflow.entities.run.Run
  :return: Spark DataFrame
  """
  delta_path = run.data.params["delta_path"]
  delta_version = run.data.params["delta_version"]
  print(f"Loading Delta table from path: {delta_path}; version: {delta_version}")
  df = spark.read.format("delta").option("versionAsOf", delta_version).load(delta_path)
  
  return df  

# COMMAND ----------

"""
MLflow Registry Utility Methods
"""

def transition_model(model_version, stage):
    """
    Transition a model to a specified stage in MLflow Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    :param model_version: mlflow.entities.model_registry.ModelVersion. ModelVersion object to transition
    :param stage: (str) New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"

    :return: A single mlflow.entities.model_registry.ModelVersion object
    """
    client = MlflowClient()
    
    model_version = client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )

    return model_version  
  

def fetch_model_version(registry_model_name, stage="Staging"):
    """
    For a given registered model, return the MLflow ModelVersion object
    This contains all metadata needed, such as params logged etc

    :param registry_model_name: (str) Name of MLflow Registry Model
    :param stage: (str) Stage for this model. One of "Staging" or "Production"

    :return: mlflow.entities.model_registry.ModelVersion
    """
    client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_model = client.search_registered_models(filter_string=filter_string)[0]

    if len(registered_model.latest_versions) == 1:
        model_version = registered_model.latest_versions[0]

    else:
        model_version = [model_version for model_version in registered_model.latest_versions if model_version.current_stage == stage][0]

    return model_version

  
def get_run_from_registered_model(registry_model_name, stage="Staging"):
    """
    Get Mlflow run object from registered model

    :param registry_model_name: (str) Name of MLflow Registry Model
    :param stage: (str) Stage for this model. One of "Staging" or "Production"

    :return: mlflow.entities.run.Run
    """
    model_version = fetch_model_version(registry_model_name, stage)
    run_id = model_version.run_id
    run = mlflow.get_run(run_id)

    return run  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training Functions

# COMMAND ----------

def create_sklearn_rf_pipeline(model_params, seed=42):
  """
  Create the sklearn pipeline required for the RandomForestRegressor.
  We compose two components of the pipeline separately - one for numeric cols, one for categorical cols
  These are then combined with the final RandomForestRegressor stage, which uses the model_params dict
  provided via the args. The unfitted pipeline is returned.
  
  For a robust pipeline in practice, one should also have a pipeline stage to add indicator columns for those features
  which have been imputed. This can be useful to encode information about those instances which have been imputed with
  a given value. We refrain from doing so here to simplify the pipeline, and focus on the overall workflow.
  
  :param model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor
  :param seed : (int) Random seed to set via random_state arg in RandomForestRegressor
 
  :return: sklearn pipeline
  """
  # Create pipeline component for numeric Features
  numeric_transformer = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy='median'))])

  # Create pipeline component for categorical Features
  categorical_transformer = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("ohe", OneHotEncoder(handle_unknown="ignore"))])

  # Combine numeric and categorical components into one preprocessor pipeline
  # Use ColumnTransformer to apply the different preprocessing pipelines to different subsets of features
  # Use selector (make_column_selector) to select which subset of features to apply pipeline to
  preprocessor = ColumnTransformer(transformers=[
      ("numeric", numeric_transformer, selector(dtype_exclude="category")),
      ("categorical", categorical_transformer, selector(dtype_include="category"))
  ])

  pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                             ("rf", RandomForestRegressor(random_state=seed, 
                                                          **model_params))
                            ])
  
  return pipeline


def train_sklearn_rf_model(run_name, delta_path, model_params, misc_params, seed=42):
  """
  Function to trigger training and evaluation of an sklearn RandomForestRegressor model.
  Parameters, metrics and artifacts are logged to MLflow during this process.
  Return the MLflow run object 
  
  :param run_name: (str) name to give to MLflow run
  :param delta_path: (str) path to Delta table to use as input data
  :param model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor
  :param misc_params: (dict) Dictionary of params to use 
  
  :return: mlflow.entities.run.Run  
  """  
  with mlflow.start_run(run_name=run_name) as run:

    # Enable MLflow autologging
    mlflow.autolog(log_input_examples=True, silent=True)
    
    # Load Delta table from delta_path
    df = spark.read.format("delta").load(delta_path)   
    # Log Delta path and version
    mlflow.log_param("delta_path", delta_path)
    delta_version = get_delta_version(delta_path)
    mlflow.log_param("delta_version", delta_version)
    
    # Track misc parameters used in pipeline creation (preprocessing) as json artifact
    mlflow.log_dict(misc_params, "preprocessing_params.json")
    target_col = misc_params["target_col"]  
    num_cols = misc_params["num_cols"]    
    cat_cols = misc_params["cat_cols"]    

    # Convert Spark DataFrame to pandas, as we will be training an sklearn model
    pdf = df.toPandas() 
    # Convert all cat cols to category dtype
    for c in cat_cols:
        pdf[c] = pdf[c].astype("category")    
    
    # Create summary statistics pandas DataFrame and log as a csv to MLflow
    summary_stats_pdf = create_summary_stats_pdf(pdf)
    log_summary_stats_pdf_as_csv(summary_stats_pdf)  
    
    # Track number of total instances and "month"
    num_instances = pdf.shape[0]
    mlflow.log_param("num_instances", num_instances)  # Log number of instances
    mlflow.log_param("month", misc_params["month"])   # Log month number
    
    # Split data
    X = pdf.drop([misc_params["target_col"], "month"], axis=1)
    y = pdf[misc_params["target_col"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Track train/test data info as params
    num_training = X_train.shape[0]
    mlflow.log_param("num_training_instances", num_training)
    num_test = X_test.shape[0]
    mlflow.log_param("num_test_instances", num_test)

    # Fit sklearn pipeline with RandomForestRegressor model
    rf_pipeline = create_sklearn_rf_pipeline(model_params)
    rf_pipeline.fit(X_train, y_train)
    # Specify data schema which the model will use as its ModelSignature
    input_schema = Schema([
      ColSpec("integer", "accommodates"),
      ColSpec("integer", "bedrooms"),
      ColSpec("integer", "beds"),
      ColSpec("integer", "number_of_reviews"),
      ColSpec("integer", "number_of_reviews_ltm"),
      ColSpec("integer", "minimum_nights"),
      ColSpec("integer", "review_scores_rating"),
      ColSpec("string", "host_is_superhost"),
      ColSpec("string", "neighbourhood_cleansed"),
      ColSpec("string", "property_type"),
      ColSpec("string", "room_type")
    ])
    output_schema = Schema([ColSpec("double")])
    signature = ModelSignature(input_schema, output_schema)
    mlflow.sklearn.log_model(rf_pipeline, "model", signature=signature)

    # Evaluate the model
    predictions = rf_pipeline.predict(X_test)
    test_mse = mean_squared_error(y_test, predictions) 
    r2 = r2_score(y_test, predictions)
    mlflow.log_metrics({"test_mse": test_mse,
                       "test_r2": r2})

  return run
