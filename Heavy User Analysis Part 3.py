# Databricks notebook source
# MAGIC %md
# MAGIC # Analysis of Heavy Users - Part 3

# COMMAND ----------

# MAGIC %md
# MAGIC <a href="$./Heavy User Analysis Outliers and Oddities">Link to previous notebook of Heavy User EDA</a>

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as st
from sklearn import preprocessing
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

# COMMAND ----------

#Global variables
sample_id = 42
week_1_start = '20180923'
week_1_end = '20180929'
week_4_end = '20181020'
prob = (0.5, 0.25, 0.5, 0.75, 0.95)
ninetyfive_ind = 4 #index for 95% quantile
relError = 0

# COMMAND ----------

# MAGIC %md
# MAGIC Research/Verify search count range  
# MAGIC Per Ben, use search clients daily instead of main_summary for search counts

# COMMAND ----------

ms_1week_sc = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      search_counts
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    """.format(week_1_start, week_1_end, sample_id)
    )

# COMMAND ----------

# explode to get Search structure
ms_1week_sc=ms_1week_sc.withColumn('exploded', F.explode('search_counts'))

display(ms_1week_sc)

# COMMAND ----------

newdf_count = ms_1week_sc.where("client_id ='66fa7aa4-cafe-46e5-8ae2-5e7a6046eab7'")
display(newdf_count)

# COMMAND ----------

# get the count in a separate column
ms_1week_sc=ms_1week_sc.withColumn('search_count', F.col('exploded').getItem("count"))
display(ms_1week_sc)

# COMMAND ----------

newdf_count = ms_1week_sc.where("client_id ='66fa7aa4-cafe-46e5-8ae2-5e7a6046eab7'")
display(newdf_count)

# COMMAND ----------

ms_1week_sc=ms_1week_sc.drop('search_counts', 'exploded')
display(ms_1week_sc)

# COMMAND ----------

newdf_count = ms_1week_sc.where("client_id ='66fa7aa4-cafe-46e5-8ae2-5e7a6046eab7'")
display(newdf_count)

# COMMAND ----------

# Sum over the day
ms_1week_sc_sum = ms_1week_sc.groupby('client_id', 'submission_date_s3').sum()
display(ms_1week_sc_sum)

# COMMAND ----------

# Average over the week
ms_1week_sc_avg = ms_1week_sc_sum.groupby('client_id').avg().withColumnRenamed('avg(sum(search_count))','avg_search_counts')
display(ms_1week_sc_sum)

# COMMAND ----------

newdf_count = ms_1week_sc_sum.where("client_id ='66fa7aa4-cafe-46e5-8ae2-5e7a6046eab7'")
display(newdf_count)

# COMMAND ----------

ms_1week_sc_avg = ms_1week_sc_sum.groupby('client_id').avg().withColumnRenamed('avg(sum(search_count))','avg_search_counts')
display(ms_1week_sc_avg)

# COMMAND ----------

newdf_count = ms_1week_sc_avg.where("client_id ='66fa7aa4-cafe-46e5-8ae2-5e7a6046eab7'")
display(newdf_count)

# COMMAND ----------

# MAGIC %md
# MAGIC Research pings with 0 uri, 0 active_ticks and high subsession hours

# COMMAND ----------

# Get all the data from main_summary for 4 weeks
ms_1week_raw = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      coalesce(scalar_parent_browser_engagement_total_uri_count, 0) AS uri,
      (coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600 AS active_tick_hrs,
      subsession_length/3600 AS subsession_hours
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    ORDER BY
      1, 2
    """.format(week_1_start, week_1_end, sample_id)
    )


# COMMAND ----------

ms_1week_raw.describe().show()

# COMMAND ----------

ms_1week_zero_uri = ms_1week_raw.filter((F.col("uri") == 0))
ms_1week_zero_uri.describe().show()

# COMMAND ----------

newdf_count = ms_1week_raw.where("client_id ='7e6c052f-3875-4f34-ad34-dfe4c808c903'")
display(newdf_count)

# COMMAND ----------

display(ms_1week_raw)

# COMMAND ----------

ms_1week_0_uri_at = ms_1week_raw.filter((F.col("uri") == 0) & (F.col("active_tick_hrs") == 0))
display(ms_1week_0_uri_at)

# COMMAND ----------

ms_1week_0_sh = ms_1week_raw.filter((F.col("subsession_hours") == 0) & ((F.col("active_tick_hrs") > 0) | (F.col("uri") > 0)))
display(ms_1week_0_sh)

# COMMAND ----------

ms_1week_0_sh.describe().show()

# COMMAND ----------

ms_1week_0_sh_uri = ms_1week_raw.filter((F.col("subsession_hours") == 0) & (F.col("uri") > 0))
display(ms_1week_0_sh_uri)

# COMMAND ----------

ms_1week_0_sh_uri.describe().show()

# COMMAND ----------

ms_1week_high_sh = ms_1week_raw.filter((F.col("subsession_hours") >24))
ms_1week_high_sh.describe().show()

# COMMAND ----------

display(ms_1week_high_sh)

# COMMAND ----------

ms_1week_low_sh = ms_1week_raw.filter((F.col("subsession_hours") <24))
ms_1week_low_sh.describe().show()

# COMMAND ----------



# COMMAND ----------


