# Databricks notebook source
# MAGIC %md
# MAGIC # Inventory workspace model registry entities
# MAGIC
# MAGIC This notebook uses the model registry REST API to copy all registered model and model version metadata into Delta tables. You can use then use the Delta tables to understand your model registry entities and select entities to delete.

# COMMAND ----------

# MAGIC %pip install numpy pandas tqdm mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import math
import numpy as np
import pandas as pd
import requests

DATABRICKS_WORKSPACE = # put your workspace URL here, e.g. "https://ml_team.cloud.databricks.com"
TOKEN = # put your personal access token here
HEADERS = {'Authorization': f'Bearer {TOKEN}'}
SEARCH_REGISTERED_MODELS_ENDPOINT = f"{DATABRICKS_WORKSPACE}/api/2.0/mlflow/registered-models/search"
SEARCH_MODEL_VERSIONS_ENDPOINT = f"{DATABRICKS_WORKSPACE}/api/2.0/mlflow/model-versions/search"

REGISTERED_MODEL_TABLE = # define the name of the table to store the registered models, e.g. "ml_team.default.workspace_registered_models"
MODEL_VERSION_TABLE = # define the name of the table to store the model versions, e.g. "ml_team.default.workspace_model_versions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registered models
# MAGIC
# MAGIC The following code lists all registered models in a workspace and copies their metadata into a Delta table.

# COMMAND ----------

def transform_registered_model(model: dict) -> dict:
    latest_versions = model.get("latest_versions", [])
    latest_version_number = max(mv.get("version") for mv in latest_versions) if latest_versions else np.nan
    prod_version_number = next((mv['version'] for mv in latest_versions if mv.get('current_stage') == 'Production'), np.nan)
    staging_version_number = next((mv['version'] for mv in latest_versions if mv.get('current_stage') == 'Staging'), np.nan)
    return {
        "name": model.get("name"),
        "description": model.get("description", np.nan),
        "user_id": model.get("user_id"),
        "creation_timestamp": model.get("creation_timestamp"),
        "last_updated_timestamp": model.get("last_updated_timestamp"),
        "latest_version_number": latest_version_number,
        "latest_prod_version_number": prod_version_number,
        "latest_staging_version_number": staging_version_number,
        "tags": model.get("tags", np.nan)
    }

# COMMAND ----------

registered_models = []

search_params = {
    "filter": "",
    "max_results": 1000, # max results per page
    "order_by": ["name ASC"]
}

def get_registered_models(search_params):
    backoff_exponent = 1
    max_retries = 10
    retries = 0
    while retries < max_retries:
        response = requests.get(SEARCH_REGISTERED_MODELS_ENDPOINT, headers=HEADERS, params=search_params)
        if response.status_code == 200:
            time.sleep(0.5)  # respect a 2 QPS limit, so that UI-based model search is not impacted.
            return response.json()
        elif response.status_code == 429 or response.status_code >= 500:
            time_to_sleep = min(math.exp(backoff_exponent), 30) # exponential backoff with cap at 30 seconds
            print(f"Rate limited or internal error. Retrying in {time_to_sleep} seconds...")
            time.sleep(time_to_sleep)
            backoff_exponent += 1
        else:
            print("Unexpected error: {}. Exiting...".format(response.text))
            break
        retries += 1
    return None

data = get_registered_models(search_params)
if data:
    registered_models = [transform_registered_model(model) for model in data.get('registered_models', [])]

    # Handle pagination if there are more models to fetch
    next_page_token = data.get('next_page_token', None)
    while next_page_token:
        search_params['page_token'] = next_page_token
        data = get_registered_models(search_params)
        if data:
            registered_models.extend([transform_registered_model(model) for model in data.get('registered_models', [])])
            next_page_token = data.get('next_page_token', None)
        else:
            break

# Convert the list of registered models to a DataFrame
df_registered_models = pd.json_normalize(registered_models)
df_registered_models.head()


# COMMAND ----------

# Write the registered models to the REGISTERED_MODEL_TABLE
spark_df = spark.createDataFrame(df_registered_models)
spark.sql("DROP TABLE IF EXISTS {}".format(REGISTERED_MODEL_TABLE))
spark_df.write.saveAsTable(REGISTERED_MODEL_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model versions
# MAGIC
# MAGIC The following code lists the latest 1000 model versions of each registered model in a workspace and copies their metadata into a Delta table.
# MAGIC
# MAGIC Note: The `search_model_versions` REST API does not support pagination, nor does it support filtering based on the version column. To list all model versions under a registered model, Databricks recommends listing model versions 10,000 at a time in order of descending version, deleting versions you no longer need, and then repeating the process as necessary until you are under the limit.
# MAGIC
# MAGIC ```
# MAGIC search_params = {
# MAGIC         "filter": f"name = 'too_many_model_versions'",
# MAGIC         "max_results": 10000,
# MAGIC         "order_by": ["version DESC"]
# MAGIC }
# MAGIC ```

# COMMAND ----------

from tqdm import tqdm 

model_versions = []

def get_model_versions(model_name):
    # Make search call
    search_params = {
        "filter": "name = '{}'".format(model_name.replace("'", "\\'")),
        # since model version search is unpaginated, we list 1000 model versions for each registered model
        "max_results": 1000,
        "order_by": ["version DESC"]
    }

    backoff_exponent = 1
    max_retries = 10
    retries = 0
    while retries < max_retries:
        response = requests.get(SEARCH_MODEL_VERSIONS_ENDPOINT, headers=HEADERS, params=search_params)
        if response.status_code == 200:
            time.sleep(0.025)  # respect the 40 QPS limit
            return response.json()
        elif response.status_code == 429 or response.status_code >= 500:
            time_to_sleep = min(math.exp(backoff_exponent), 30) # exponential backoff with cap at 30 seconds
            print(f"Rate limited or internal error. Retrying in {time_to_sleep} seconds...")
            time.sleep(time_to_sleep)
            backoff_exponent += 1
        else:
            print("Unexpected error fetching model versions for model: {}".format(model_name))
            print("Error: {}".format(response.text))
            print("Skipping model: {}".format(model_name))
            break
        retries += 1
    return None

for model in tqdm(registered_models, desc="Fetching model versions"):
    model_name = model.get("name")
    data = get_model_versions(model_name)
    if data and 'model_versions' in data:
        model_versions.extend(data['model_versions'])

# Convert the list of model versions to a DataFrame
df_model_versions = pd.json_normalize(model_versions)
df_model_versions.head()

# COMMAND ----------

# Write the model versions to MODEL_VERSION_TABLE
spark_df = spark.createDataFrame(df_model_versions)
spark.sql("DROP TABLE IF EXISTS {}".format(MODEL_VERSION_TABLE))
spark_df.write.saveAsTable(MODEL_VERSION_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete unused entities
# MAGIC
# MAGIC Now, run SQL queries on the registered model and model version metadata you've stored in Delta Tables. Use SQL queries to identify and delete unused entities.

# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()

# COMMAND ----------

# Select all registered models owned by a specific user
USER_ID = 'some_user@test.com'
to_delete_rm_sdf = spark.sql("SELECT * FROM {} WHERE user_id = '{}'".format(REGISTERED_MODEL_TABLE, USER_ID))
to_delete_rm_sdf.display()

# COMMAND ----------

for row in tqdm(to_delete_rm_sdf.collect(), desc="Deleting registered models"):
    client.delete_registered_model(name=row['name']) # has 429 retries built-in
    time.sleep(0.025)  # respect the 40 QPS limit

# COMMAND ----------

# Select all model versions in a registered model that are not in Production or Staging stages
MY_MODEL_NAME = 'my_test_model'
to_delete_mv_sdf = spark.sql("SELECT * FROM {} WHERE name = '{}' and current_stage NOT IN ('Production', 'Staging')".format(MODEL_VERSION_TABLE, MY_MODEL_NAME))
to_delete_mv_sdf.display()

# COMMAND ----------

for row in tqdm(to_delete_mv_sdf.collect(), desc="Deleting model versions"):
    client.delete_model_version(name=row['name'], version=row['version']) # has 429 retries built-in
    time.sleep(0.025)  # respect the 40 QPS limit
