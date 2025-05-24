# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #### Monitoring Utility Functions
# MAGIC
# MAGIC The following functions check
# MAGIC - the proportion of nulls
# MAGIC - the differences in summary statistics
# MAGIC - the shifts in distributions

# COMMAND ----------

def check_null_proportion(new_pdf, null_proportion_threshold):
  """
  Function to compute the proportions of nulls for all columns in Spark DataFrame and return any features that exceed the specified null threshold.
  
  :param df: (pd.DataFrame) The dataframe that contains new incoming data
  :param null_proportion_threshold: (float) A numeric value ranging from 0 and 1 that specifies the tolerable fraction of nulls. 
  """
  missing_stats = pd.DataFrame(new_pdf.isnull().sum() / len(new_pdf)).transpose()
  null_dict = {}
  null_col_list = missing_stats.columns[(missing_stats >= null_proportion_threshold).iloc[0]]
  for feature in null_col_list:
    null_dict[feature] = missing_stats[feature][0]
  try:
    assert len(null_dict) == 0
  except:
    print("Alert: There are feature(s) that exceed(s) the expected null threshold. Please ensure that the data is ingested correctly")
    print(null_dict)

# COMMAND ----------

def check_diff_in_summary_stats(new_stats_pdf, prod_stats_pdf, num_cols, stats_threshold_limit, statistic_list):
  """
  Function to check if the new summary stats significantly deviates from the summary stats in the production data by a certain threshold. 
  
  :param new_stats_pdf: (pd.DataFrame) summary statistics of incoming data
  :param prod_stats_pdf: (pd.DataFrame) summary statistics of production data
  :param num_cols: (list) a list of numeric columns
  :param stats_threshold_limit: (double) a float < 1 that signifies the threshold limit
  :param compare_stats_name: (string) can be one of mean, std, min, max
  :param feature_diff_list: (list) an empty list to store the feature names with differences
  """ 
  feature_diff_list = []
  for feature in num_cols: 
    print(f"\nCHECKING {feature}.........")
    for statistic in statistic_list: 
      val = prod_stats_pdf[[str(feature)]].loc[str(statistic)][0]
      upper_val_limit = val * (1 + stats_threshold_limit)
      lower_val_limit = val * (1 - stats_threshold_limit)

      new_metric_value = new_stats_pdf[[str(feature)]].loc[str(statistic)][0]

      if new_metric_value < lower_val_limit:
        feature_diff_list.append(str(feature))
        print(f"\tThe {statistic} {feature} in the new data is at least {stats_threshold_limit *100}% lower than the {statistic} in the production data. Decreased from {round(val, 2)} to {round(new_metric_value,2)}.")

      elif new_metric_value > upper_val_limit:
        feature_diff_list.append(str(feature))
        print(f"\tThe {statistic} {feature} in the new data is at least {stats_threshold_limit *100}% higher than the {statistic} in the production data. Increased from {round(val, 2)} to {round(new_metric_value, 2)}.")

      else:
        pass
  
  return np.unique(feature_diff_list)

# COMMAND ----------

def check_diff_in_variances(reference_df, new_df, num_cols, p_threshold):
  """
  This function uses the Levene test to check if each column's variance in new_df is significantly different from reference_df
  From docs: The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.
  
  :param reference_df(pd.DataFrame): current dataframe in production
  :param new_df (pd.DataFrame): new dataframe
  :param num_cols (list): a list of numeric features
  
  ‘median’ : Recommended for skewed (non-normal) distributions.
  """
  var_dict = {}
  for feature in num_cols:
    levene_stat, levene_pval = stats.levene(reference_df[str(feature)], new_df[str(feature)], center="median")
    if levene_pval <= p_threshold:
      var_dict[str(feature)] = levene_pval
  try:
    assert len(var_dict) == 0
    print(f"No features have significantly different variances compared to production data at p-value {p_threshold}")
  except:
    print(f"The feature(s) below have significantly different variances compared to production data at p-value {p_threshold}")
    print(var_dict)

# COMMAND ----------

def check_dist_ks_bonferroni_test(reference_df, new_df, num_cols, p_threshold, ks_alternative="two-sided"):
    """
    Function to take two pandas DataFrames and compute the Kolmogorov-Smirnov statistic on 2 sample distributions
    where the variable in question is continuous.
    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous
    distribution. If the KS statistic is small or the p-value is high, then we cannot reject the hypothesis that 
    the distributions of the two samples are the same.
    The alternative hypothesis can be either ‘two-sided’ (default), ‘less’ or ‘greater’.
    This function assumes that the distributions to compare have the same column name in both DataFrames.
    
    see more details here: https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test

    :param reference_df: pandas DataFrame containing column with the distribution to be compared
    :param new_df: pandas DataFrame containing column with the distribution to be compared
    :param col_name: (str) Name of colummn to use as variable to create numpy array for comparison
    :param ks_alternative: Defines the alternative hypothesis - ‘two-sided’ (default), ‘less’ or ‘greater’.
    """
    ks_dict = {}
    ### Bonferroni correction 
    corrected_alpha = p_threshold / len(num_cols)
    print(f"The Bonferroni-corrected alpha level is {round(corrected_alpha, 4)}. Any features with KS statistic below this alpha level have shifted significantly.")
    for feature in num_cols:
      ks_stat, ks_pval = stats.ks_2samp(reference_df[feature], new_df[feature], alternative=ks_alternative, mode="asymp")
      if ks_pval <= corrected_alpha:
        ks_dict[feature] = ks_pval
    try:
      assert len(ks_dict) == 0
      print(f"No feature distributions has shifted according to the KS test at the Bonferroni-corrected alpha level of {round(corrected_alpha, 4)}. ")
    except:
      print(f"The feature(s) below have significantly different distributions compared to production data at Bonferroni-corrected alpha level of {round(corrected_alpha, 4)}, according to the KS test")
      print("\t", ks_dict)
      

# COMMAND ----------

def check_categorical_diffs(reference_pdf, new_pdf, cat_cols, p_threshold):
  """
  This function checks if there are differences in expected counts for categorical variables between the incoming data and the data in production.
  
  :param new_pdf: (pandas DataFrame) new incoming data
  :param reference_pdf: (pandas DataFrame) data in production
  :param cat_cols: (list) a list of categorical columns
  """
  chi_dict = {}
  catdiff_list = []
  
  # Compute modes for all cat cols
  reference_modes_pdf = reference_pdf[cat_cols].mode(axis=0, numeric_only=False, dropna=True)
  new_modes_pdf = new_pdf[cat_cols].mode(axis=0, numeric_only=False, dropna=True)
  
  for feature in cat_cols: 
    prod_array = reference_pdf[feature].value_counts(ascending=True).to_numpy()
    new_array = new_pdf[feature].value_counts(ascending=True).to_numpy()
    try:
      chi_stats, chi_pval = stats.chisquare(new_array, prod_array)
      if chi_pval <= p_threshold:
        chi_dict[feature] = chi_pval
    except ValueError as ve :
      catdiff_list.append(feature)
      
    # Check if the mode has changed
    
    reference_mode = reference_modes_pdf[feature].iloc[0]
    new_mode = new_modes_pdf[feature].iloc[0]
    try:
      assert reference_mode == new_mode
    except:
      print(f"The mode for {feature} has changed from {reference_mode} to {new_mode}.")

  print(f"\nCategorical varibles with different number of levels compared to the production data:")
  print("\t", catdiff_list)
  print(f"\nChi-square test with p-value of {p_threshold}:")
  print(f"\tCategorical variables with significantly different expected count: {chi_dict}")


# COMMAND ----------

def compare_model_perfs(current_staging_run, current_prod_run, min_model_perf_threshold, metric_to_check):
  """
  This model compares the performances of the models in staging and in production. 
  Outputs: Recommendation to transition the staging model to production or not
  
  :param current_staging_run: MLflow run that contains information on the staging model
  :param current_prod_run: MLflow run that contains information on the production model
  :param min_model_perf_threshold (float): The minimum threshold that the staging model should exceed before being transitioned to production
  :param metric_to_check (string): The metric that the user is interested in using to compare model performances
  """
  model_diff_fraction = current_staging_run.data.metrics[str(metric_to_check)] / current_prod_run.data.metrics[str(metric_to_check)]
  model_diff_percent = round((model_diff_fraction - 1)*100, 2)
  print(f"Staging run's {metric_to_check}: {round(current_staging_run.data.metrics[str(metric_to_check)],3)}")
  print(f"Current production run's {metric_to_check}: {round(current_prod_run.data.metrics[str(metric_to_check)],3)}")

  if model_diff_percent >= 0 and (model_diff_fraction - 1 >= min_model_perf_threshold):
    print(f"The current staging run exceeds the model improvement threshold of at least +{min_model_perf_threshold}. You may proceed with transitioning the staging model to production now.")
    
  elif model_diff_percent >= 0 and (model_diff_fraction - 1  < min_model_perf_threshold):
    print(f"CAUTION: The current staging run does not meet the improvement threshold of at least +{min_model_perf_threshold}. Transition the staging model to production with caution.")
  else: 
    print(f"ALERT: The current staging run underperforms by {model_diff_percent}% when compared to the production model. Do not transition the staging model to production.")

# COMMAND ----------

def plot_boxplots(unique_feature_diff_array, reference_pdf, new_pdf):
  sns.set_theme(style="whitegrid")
  fig, ax = plt.subplots(len(unique_feature_diff_array), 2, figsize=(15,8))
  fig.suptitle("Distribution Comparisons between Incoming Data and Production Data")
  ax[0, 0].set_title("Production Data")
  ax[0, 1].set_title("Incoming Data")

  for i in range(len(unique_feature_diff_array)):
    p1 = sns.boxplot(ax=ax[i, 0], x=reference_pdf[unique_feature_diff_array[i]])
    p1.set_xlabel(str(unique_feature_diff_array[i]))
    p1.annotate(str(unique_feature_diff_array[i]), xy=(10,0.5))
    p2 = sns.boxplot(ax=ax[i, 1], x=new_pdf[unique_feature_diff_array[i]])
    p2.annotate(str(unique_feature_diff_array[i]), xy=(10,0.5))
