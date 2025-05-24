# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Data+AI Summit 2021
# MAGIC
# MAGIC #### Drifting Away: Testing ML Models in Production
# MAGIC
# MAGIC The following notebook is an accompaniment to the Data+AI Summit talk, [Drifting Away: Testing ML Models in Production](https://databricks.com/session_na21/drifting-away-testing-ml-models-in-production).
# MAGIC
# MAGIC *Notebooks available here:* [bit.ly/dais_2021_drifting_away](https://drive.google.com/drive/folders/1-uMqTjsU2I7ODLq5wUpAdBHFkNjvZDVD)
# MAGIC
# MAGIC **Data Citation:**
# MAGIC * *[Inside Airbnb - Hawaii Listings Dataset](http://insideairbnb.com/get-the-data.html)*
# MAGIC
# MAGIC
# MAGIC **Requirements**
# MAGIC * The following notebook was developed and tested using [DBR 8.2 ML](https://docs.databricks.com/release-notes/runtime/8.2ml.html)
# MAGIC * Additional dependencies:
# MAGIC   * `mlflow==1.16.0`
# MAGIC   * `scipy==1.6.3`
# MAGIC   * `seaborn==0.11.1`
# MAGIC   
# MAGIC **Authors**
# MAGIC - Chengyin Eng
# MAGIC - Niall Turbitt 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outline
# MAGIC
# MAGIC We simulate a batch inference scenario where we train, deploy, and maintain a model to predict Airbnb house listings in Hawaii on a monthly basis. 
# MAGIC
# MAGIC **Data interval**: Arrives monthly <br>
# MAGIC
# MAGIC **Workflow**: 
# MAGIC * Load the new month of incoming data
# MAGIC * Apply incoming data checks 
# MAGIC   * Error and drift evaluation
# MAGIC * Identify and address any errors and drifts
# MAGIC * Train a new model
# MAGIC * Apply model validation checks versus the existing model in production
# MAGIC     * If checks pass, deploy the new candidate model to production
# MAGIC     * If checks fail, do not deploy the new candidate model <br>
# MAGIC     
# MAGIC **Reproducibility Tools**: 
# MAGIC * [MLflow](https://www.mlflow.org/docs/latest/index.html) for model parameters, metrics, and artifacts
# MAGIC * [Delta](https://docs.delta.io/latest/index.html) for data versioning <br>
# MAGIC
# MAGIC Although this notebook specifically addresses tests to monitor a supervised ML model for batch inference, the same tests are applicable in streaming and real-time settings.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Run setup and utils notebooks

# COMMAND ----------



# COMMAND ----------

# MAGIC %run ./training_setup

# COMMAND ----------

# MAGIC %run ./monitoring_utils

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("stats_threshold_limit", "0.5")
dbutils.widgets.text("p_threshold", "0.05")
dbutils.widgets.text("min_model_r2_threshold", "0.005")

stats_threshold_limit = float(dbutils.widgets.get("stats_threshold_limit"))       # how much we should allow basic summary stats to shift 
p_threshold = float(dbutils.widgets.get("p_threshold"))                           # the p-value below which to reject null hypothesis 
min_model_r2_threshold = float(dbutils.widgets.get("min_model_r2_threshold"))     # minimum model improvement

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Month 0
# MAGIC
# MAGIC * Train an inital model to predict listing prices and deploy to Production
# MAGIC * As we have no historic data or existing models in production to compare against, we do not apply error and distribution checks here.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### i. Initial Data load
# MAGIC
# MAGIC Load the first month of data which we use to train and evaluate our first model. 
# MAGIC
# MAGIC We create a "Gold" table to which we will be appending each subsequent month of data.

# COMMAND ----------

# Define the path to use for our Gold table
gold_delta_path = data_project_dir + "airbnb_hawaii_delta"

# Ensure we start with no existing Delta table 
dbutils.fs.rm(gold_delta_path, True)

# Incoming Month 0 Data
month_0_df = spark.read.format("delta").load(month_0_delta_path)

# Create inital version of the Gold Delta table we will use for training - this will be updated with subsequent "months" of data
month_0_df.withColumn("month", F.lit("month_0")).write.format("delta").partitionBy("month").save(gold_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training

# COMMAND ----------

# Set the month number - used for naming the MLflow run and tracked as a parameter 
month = 0

# Specify name of MLflow run
run_name = f"month_{month}"

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}

# Trigger model training and logging to MLflow
month_0_run = train_sklearn_rf_model(run_name, 
                                     gold_delta_path, 
                                     model_params, 
                                     misc_params)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### iii. Model Deployment
# MAGIC
# MAGIC We first register the model to the [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html). For demonstration purposes we will immediately transition the model to the "Production" stage in the MLflow Model Registry, however in a real world scenario one should have a robust model validation process in place prior to migrating a model to Production. 
# MAGIC
# MAGIC We will demonstrate a multi-stage approach in the subsequent sections, first transitioning a model to "Staging", conducting model validation checks, and only then triggering a transition from Staging to Production once these checks are satistified.

# COMMAND ----------

# Register model to MLflow Model Registry
month_0_run_id = month_0_run.info.run_id
month_0_model_version = mlflow.register_model(model_uri=f"runs:/{month_0_run_id}/model", name=registry_model_name)

# COMMAND ----------

# Transition model to Production
month_0_model_version = transition_model(month_0_model_version, stage="Production")
print(month_0_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Month 1 - New Data Arrives
# MAGIC
# MAGIC Our model has been deployed for a month and we now have an incoming fresh month of data.
# MAGIC
# MAGIC **Scenario 1:**
# MAGIC * During the upstream data cleansing process, certain neighbourhoods are missing their `neighbourhood_cleansed` entries, i.e. Primary Urban Center, Kihei-Makena, Lahaina, North Kona
# MAGIC * Also during the upstream data generation procedure a new "star" rating system between 0 and 5 for `review_scores_rating` was introduced without our knowledge.
# MAGIC   - Instances in this new month of data will have `review_scores_rating` entries scaled between 0 and 5, where historic entries were bounded between 0 and 100.
# MAGIC   
# MAGIC **What are we simulating here?**
# MAGIC * Feature drift
# MAGIC * Upstream data errors
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### i. Feature checks prior to model training
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks
# MAGIC
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

# Incoming Month 1 Data - we have synthesised some errors/distribution changes in our data that we would necessarily want to catch
# Create pandas DataFrame from Spark DataFrame 
# (note that we are doing this due to the small size of this data, however caution should be taken here for larger datasets)
month_1_err_pdf = spark.read.format("delta").load(month_1_error_delta_path).toPandas()

# Compute summary statistics on new incoming data
summary_stats_month_1_err_pdf = create_summary_stats_pdf(month_1_err_pdf)

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Production
current_prod_run_1 = get_run_from_registered_model(registry_model_name, stage="Production")

# Load in original versions of Delta table used at training time for current Production model
current_prod_pdf_1 = load_delta_table_from_run(current_prod_run_1).toPandas()

# Load summary statistics pandas DataFrame for data which the model currently in Production was trained and evaluated against
current_prod_stats_pdf_1 = load_summary_stats_pdf_from_run(current_prod_run_1, project_local_tmp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks

# COMMAND ----------

print("CHECKING PROPORTION OF NULLS.....")
check_null_proportion(month_1_err_pdf, null_proportion_threshold=.5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks

# COMMAND ----------

statistic_list = ["mean", "median", "std", "min", "max"]

# Check if the new summary stats deviate from previous summary stats by a certain threshold
unique_feature_diff_array_month_1 = check_diff_in_summary_stats(summary_stats_month_1_err_pdf, 
                                                                current_prod_stats_pdf_1, 
                                                                num_cols + [target_col], # Include the target col in this analysis
                                                                stats_threshold_limit, 
                                                                statistic_list)
unique_feature_diff_array_month_1

# COMMAND ----------

print(f"Let's look at the box plots of the features that exceed the stats_threshold_limit of {stats_threshold_limit}")
plot_boxplots(unique_feature_diff_array_month_1, current_prod_pdf_1, month_1_err_pdf)

# COMMAND ----------

print("\nCHECKING VARIANCES WITH LEVENE TEST.....")
check_diff_in_variances(current_prod_pdf_1, month_1_err_pdf, num_cols, p_threshold)

print("\nCHECKING KS TEST.....")
check_dist_ks_bonferroni_test(current_prod_pdf_1, month_1_err_pdf, num_cols, p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the KS test, only the `review_scores_rating` is shown to have shifted significantly.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

# Check that each categorical feature has the same mode and expected frequency distribution
check_categorical_diffs(current_prod_pdf_1, month_1_err_pdf, cat_cols, p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`Action`: Resolve Data Issues**
# MAGIC
# MAGIC After catching the data issues and working with the team in charge of upstream data processing, we resolve the issues with `review_scores_rating` and `neighbourhood_cleansed`. We add the fixed new month of data to our Gold Delta table and conduct training on the newly available data.

# COMMAND ----------

# Incoming Month 1 Data where upstream errors have been fixed
month_1_df = spark.read.format("delta").load(month_1_fixed_delta_path)

# Append new month of data to Gold Delta table to use for training
month_1_df.withColumn("month", F.lit("month_1")).write.format("delta").partitionBy("month").mode("append").save(gold_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training
# MAGIC
# MAGIC Retrain the same model, but this time we are able to use an extra month of data

# COMMAND ----------

# Set the month number - used for naming the MLflow run and logged as a parameter
month = 1

# Specify name of MLflow run
run_name = f"month_{month}"

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}

# Trigger model training and logging to MLflow
month_1_run = train_sklearn_rf_model(run_name, 
                                     gold_delta_path,   
                                     model_params, 
                                     misc_params)

# COMMAND ----------

# Register model to MLflow Model Registry
month_1_run_id = month_1_run.info.run_id
month_1_model_version = mlflow.register_model(model_uri=f"runs:/{month_1_run_id}/model", name=registry_model_name)

# Transition model to Staging
month_1_model_version = transition_model(month_1_model_version, stage="Staging")
print(month_1_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### iii. Model checks prior to model deployment
# MAGIC

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Staging
current_staging_run_1 = get_run_from_registered_model(registry_model_name, stage="Staging")

metric_to_check = "test_r2"
compare_model_perfs(current_staging_run_1, current_prod_run_1, min_model_r2_threshold, metric_to_check)

# COMMAND ----------

month_1_model_version = transition_model(month_1_model_version, stage="Production")
print(month_1_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Month 2 - New Data Arrives
# MAGIC
# MAGIC We have had a model in production for 2 months now and have now obtained an additional month of data.
# MAGIC
# MAGIC **Scenario 2:**
# MAGIC * The new month of data contains listing entries recorded during peak vacation season. As a result, the price for every listing has been increased.
# MAGIC
# MAGIC **What are we simulating here?**
# MAGIC * Label drift
# MAGIC * Concept drift
# MAGIC   * The underlying relationship between the features and label has changed due to seasonality

# COMMAND ----------

# MAGIC %md
# MAGIC #### i. Feature checks prior to model training
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks
# MAGIC
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

# Incoming Month 2 Data - we have synthesised some distribution changes in our label which we would necessarily want to catch
month_2_df = spark.read.format("delta").load(month_2_delta_path)

# Compute summary statistics on new incoming data
month_2_pdf = month_2_df.toPandas()
summary_stats_month_2_pdf = create_summary_stats_pdf(month_2_pdf)

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Production
current_prod_run_2 = get_run_from_registered_model(registry_model_name, stage="Production")

# Load in original versions of Delta table used at training time for current Production model
current_prod_pdf_2 = load_delta_table_from_run(current_prod_run_2).toPandas()

# Load summary statistics pandas DataFrame for data which the model currently in Production was trained and evaluated against
current_prod_stats_pdf_2 = load_summary_stats_pdf_from_run(current_prod_run_2, project_local_tmp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks

# COMMAND ----------

print("\nCHECKING PROPORTION OF NULLS.....")
check_null_proportion(month_2_pdf, null_proportion_threshold=.5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks

# COMMAND ----------

unique_feature_diff_array_month_2 = check_diff_in_summary_stats(summary_stats_month_2_pdf, 
                                                                current_prod_stats_pdf_2, 
                                                                num_cols + [target_col], 
                                                                stats_threshold_limit, 
                                                                statistic_list)

unique_feature_diff_array_month_2

# COMMAND ----------

print(f"Let's look at the box plots of the features that exceed the stats_threshold_limit of {stats_threshold_limit}")
plot_boxplots(unique_feature_diff_array_month_2, current_prod_pdf_2, month_2_pdf)

# COMMAND ----------

print("\nCHECKING VARIANCES WITH LEVENE TEST.....")
check_diff_in_variances(current_prod_pdf_2, month_2_pdf, num_cols, p_threshold)

print("\nCHECKING KS TEST.....")
check_dist_ks_bonferroni_test(current_prod_pdf_2, month_2_pdf, num_cols + [target_col], p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

check_categorical_diffs(current_prod_pdf_2, month_2_pdf, cat_cols, p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`Action`: Include new data with label drift in training**
# MAGIC
# MAGIC We observe that our label has drifted, and after analysis observe that this most recent month of data was captured during peak vacation season where Airbnb hosts have increased their listing prices. As such, we will retrain our model and include this recent month of data during training.

# COMMAND ----------

# Append the new month of data (where listings are most expensive across the board)
month_2_df.withColumn("month", F.lit("month_2")).write.format("delta").partitionBy("month").mode("append").save(gold_delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training
# MAGIC
# MAGIC Retrain the same model from previous months, including the additional month of data where the label has drifted.

# COMMAND ----------

# Set the month number - used for naming the MLflow run and logged as a parameter
month = 2

# Specify name of MLflow run
run_name = f"month_{month}"

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}

# Trigger model training and logging to MLflow
month_2_run = train_sklearn_rf_model(run_name, 
                                     gold_delta_path, 
                                     model_params, 
                                     misc_params)

# COMMAND ----------

# Register model to MLflow Model Registry
month_2_run_id = month_2_run.info.run_id
month_2_model_version = mlflow.register_model(model_uri=f"runs:/{month_2_run_id}/model", name=registry_model_name)

# Transition model to Staging
month_2_model_version = transition_model(month_2_model_version, stage="Staging")
print(month_2_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### iii. Model checks prior to model deployment

# COMMAND ----------

# Get the MLflow run associated with the model currently registered in Staging
current_staging_run_2 = get_run_from_registered_model(registry_model_name, stage="Staging")

# COMMAND ----------

metric_to_check = "test_r2"
compare_model_perfs(current_staging_run_2, current_prod_run_2, min_model_r2_threshold, metric_to_check)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In this case we note that the new candidate model in Staging performs notably worse than the current model in Production. We know from our checks prior to training that the label has drifted, and that this was due to new listing prices being recorded during vacation season. At this point we would want to prevent a migration of the new candidate model directly to Production and instead investigate if there is any way we can improve model performance. This could involve tuning the hyperparameters of our model, or additionally investigating the inclusion of additional features such as "month of the year" which could allow us to capture temporal impacts to listing prices.

# COMMAND ----------

# month_2_model_version = transition_model(month_2_model_version, stage="Production")
# print(month_2_model_version)
