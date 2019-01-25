# Databricks notebook source
# MAGIC %md
# MAGIC # Definition of Heavy Users
# MAGIC Determine a cutoff for "heavy"

# COMMAND ----------

# MAGIC %md
# MAGIC Based on analysis done by Brendan Colloran from the Strategy and Insights team in 2016, Saptarshi Guha in 2017 and Project Ahab in 2018, I will be looking at URI count, search count, subsession hours and active ticks.

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
prob = (0.05, 0.25, 0.5, 0.75, 0.95)
ninetyfive_ind = 4 #index for 95% quantile
relError = 0
outlier_prob = (0.95, -.96, 0.97, 0.98, 0.99)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC #####Read the Firefox data from main_summary for a week.  
# MAGIC Active ticks are in 5 second increments, so it is converted to hours.  
# MAGIC Subsession length is in seconds, so it is converted to hours.  
# MAGIC In looking at the difference between active_ticks and scalar_parent_browser_engagement_active_ticks, they are the same values except where active_ticks has a number scalar_parent_browser_engagement_active_ticks is often null.

# COMMAND ----------

ms_1week_sum = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)*5/3600) AS td_active_tick_hrs,
      sum(subsession_length/3600) AS td_subsession_hours
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    GROUP BY
        1, 2
    """.format(week_1_start, week_1_end, sample_id)
    )

search_wk = spark.sql("""
  SELECT client_id,
         submission_date_s3,
         engine,
         SUM(sap) as sap,
         SUM(tagged_sap) as tagged_sap,
         SUM(tagged_follow_on) as tagged_follow_on,
         SUM(organic) as in_content_organic
  FROM search_clients_daily
  WHERE
      submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
  GROUP BY
      1, 2, 3
    """.format(week_1_start, week_1_end, sample_id)
    )


# COMMAND ----------

print((ms_1week_sum.count(), len(ms_1week_sum.columns)))

# COMMAND ----------

ms_1week_sum.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Zero Counts
# MAGIC 
# MAGIC Issues: All 3 counts zeros?  
# MAGIC Uri and active ticks both zero?  
# MAGIC Subsession hours zero?  

# COMMAND ----------

ms_1week_az = ms_1week_sum.filter((F.col("td_uri") == 0)  \
                               & (F.col("td_subsession_hours") == 0) & (F.col("td_active_tick_hrs") == 0))
ms_1week_az.count()

# COMMAND ----------

display(ms_1week_az)

# COMMAND ----------

ms_1week_zero_uri_at = ms_1week_sum.filter((F.col("td_uri") == 0) & (F.col("td_active_tick_hrs") == 0))
ms_1week_zero_uri_at.describe().show()

# COMMAND ----------

display(ms_1week_zero_uri_at)

# COMMAND ----------

ms_1week_zero_uri = ms_1week_sum.filter((F.col("td_uri") == 0) & (F.col("td_active_tick_hrs") > 0))
ms_1week_zero_uri.describe().show()

# COMMAND ----------

display(ms_1week_zero_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC This would be the place to cap the erroneous large data before averaging.  
# MAGIC Cap the subsession hours to 25.  Any other caps?

# COMMAND ----------

# Replace subsession hours > 25 with 25
ms_1week_cap = ms_1week_sum.withColumn("td_subsession_hours", \
              F.when(ms_1week_sum["td_subsession_hours"] > 25, 25).otherwise(ms_1week_sum["td_subsession_hours"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Average URI, Active Ticks and Subsession Hours over the week.  This will only average over the days that client data exists - in other words, it will not average in a zero for days with no record.  
# MAGIC   
# MAGIC I'm not capping any of the values either before or after averaging.

# COMMAND ----------

ms_1week_3_avg = ms_1week_sum.groupBy('client_id').avg() \
  .withColumnRenamed('avg(td_uri)','avg_uri') \
  .withColumnRenamed('avg(td_active_tick_hrs)', 'avg_active_tick_hrs') \
  .withColumnRenamed('avg(td_subsession_hours)', 'avg_subsession_hours')

# COMMAND ----------

# MAGIC %md
# MAGIC Per Ben and DTMO, get the search counts from search_clients_daily instead of main summary.  
# MAGIC Average Search Counts over the week. This will only average over the days that client data exists - in other words, it will not average in a zero for days with no record.

# COMMAND ----------

search_wk = search_wk.na.fill(0)
# Sum over the day
search_wk_sum = search_wk.groupBy("client_id", "submission_date_s3").agg(F.sum(F.col("sap")+F.col("in_content_organic")))  \
                        .withColumnRenamed('sum((sap + in_content_organic))','sum_search_counts')
# Average over the week
search_wk_avg = search_wk_sum.groupby('client_id').avg() \
                                 .withColumnRenamed('avg(sum_search_counts)','avg_search_counts')

# COMMAND ----------

display(search_wk_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC Join the averaged search counts with the other averaged counts.  The full outer join will contain all the clients from both tables.  If there is no data in one of the tables, fill the blank value with 0.

# COMMAND ----------

ms_1week_avg = ms_1week_3_avg.join(search_wk_avg, ['client_id'], 'full_outer').na.fill(0)
ms_1week_avg.show()

# COMMAND ----------

ms_1week_avg.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC **There are a lot of very high max values which seems questionable. Is there an explanation for these high values? Should there be a cap on the values?**  
# MAGIC   
# MAGIC From the Project Ahab report: Subsession hours is subject to measurement error due to ping reporting, among other issues. For example, one user recorded 1.9M hours (216 years) in one 24-hour period. This measure should be a maximum of 25 hours-- 25 rather than 24 because there can be up to 1 hour lag in reporting. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Quantiles   
# MAGIC Outliers - Above the 95th percentile.

# COMMAND ----------

uri_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_uri", prob, relError)
uri_week_quantiles

# COMMAND ----------

uri_outliers = ms_1week_avg.filter(ms_1week_avg.avg_uri > uri_week_quantiles[ninetyfive_ind])
uri_outliers.count()

# COMMAND ----------

active_ticks_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_active_tick_hrs", prob, relError)
active_ticks_week_quantiles                           

# COMMAND ----------

active_ticks_outliers = ms_1week_avg.filter(ms_1week_avg.avg_active_tick_hrs > active_ticks_week_quantiles[ninetyfive_ind])
active_ticks_outliers.count()

# COMMAND ----------

subsession_hours_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_subsession_hours", prob, relError)
subsession_hours_week_quantiles                           

# COMMAND ----------

subsession_hours_outliers = ms_1week_avg.filter(ms_1week_avg.avg_subsession_hours > subsession_hours_week_quantiles[ninetyfive_ind])
subsession_hours_outliers.count()

# COMMAND ----------

search_counts_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_search_counts", prob, relError)
search_counts_week_quantiles       

# COMMAND ----------

search_counts_outliers = ms_1week_avg.filter(ms_1week_avg.avg_search_counts > search_counts_week_quantiles[ninetyfive_ind])
search_counts_outliers.count()

# COMMAND ----------

display(search_counts_outliers)

# COMMAND ----------

search_counts_outliers.describe().show()

# COMMAND ----------

newdf_uri = uri_outliers.where("client_id ='07d6a550-3d3c-4f37-b45d-204a8fe476de'")
display(newdf_uri)

# COMMAND ----------

newdf_at = active_ticks_outliers.where("client_id ='07d6a550-3d3c-4f37-b45d-204a8fe476de'")
display(newdf_at)

# COMMAND ----------

newdf_sc = search_counts_outliers.where("client_id ='07d6a550-3d3c-4f37-b45d-204a8fe476de'")
display(newdf_sc)

# COMMAND ----------

newdf_sh = subsession_hours_outliers.where("client_id ='07d6a550-3d3c-4f37-b45d-204a8fe476de'")
display(newdf_sh)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC How many rows are outliers in all 4 measurements?

# COMMAND ----------

all_outliers = uri_outliers.join(active_ticks_outliers, ['client_id'], "left_semi") \
                              .join(subsession_hours_outliers, ['client_id'], "left_semi") \
                              .join(search_counts_outliers, ['client_id'], "left_semi")

# COMMAND ----------

# MAGIC %md
# MAGIC Don't use a join, just filter on all 4 values

# COMMAND ----------

all_outliers = ms_1week_avg.filter((F.col("avg_search_counts") > search_counts_week_quantiles[ninetyfive_ind]) \
                     & (F.col("avg_active_tick_hrs") > active_ticks_week_quantiles[ninetyfive_ind]) \
                     & (F.col("avg_subsession_hours") > subsession_hours_week_quantiles[ninetyfive_ind]) \
                     & (F.col("avg_uri") > uri_week_quantiles[ninetyfive_ind] ) )
all_outliers.count()

# COMMAND ----------

display(all_outliers)

# COMMAND ----------

newdf_all = all_outliers.where("'avg_search_counts' == 1")
display(newdf_all)

# COMMAND ----------

all_outliers.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Visualizations  

# COMMAND ----------

# MAGIC %md
# MAGIC ###Histograms and Quantiles

# COMMAND ----------

# MAGIC %md
# MAGIC From the Project Ahab report for URI count: The distribution of URI count is extremely skewed with a long right-tail, so the analysis is displayed using the natural log of the weekly average of daily URI count.  

# COMMAND ----------

ms_1week_logs = ms_1week_avg.withColumn("uri_log", F.log1p("avg_uri")) \
                            .withColumn("active_tick_hrs_log", F.log1p("avg_active_tick_hrs")) \
                            .withColumn("subsession_hours_log", F.log1p("avg_subsession_hours")) \
                            .withColumn("search_counts_log", F.log1p("avg_search_counts"))

# COMMAND ----------

ms_1week_logs.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC URI distribution

# COMMAND ----------

display(ms_1week_avg.limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC Log transformed URI distribution

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC URI Quantiles  
# MAGIC The Quantiles graph is based on the first 1000 rows, so it may not be accurate.  After checking, the values are quite close to the approxQuantile 75% values

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC Search Counts  
# MAGIC  

# COMMAND ----------

display(ms_1week_avg.limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC Search Counts  
# MAGIC From the Project Ahab report for search count: The natural log transformation was applied after averaging. 

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC Search Counts Quantiles

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC Active Tick Hours

# COMMAND ----------

display(ms_1week_avg.limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC Active Tick Hours Log transformed

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC Active Tick Hours Quantiles

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC Subsession Hours

# COMMAND ----------

display(ms_1week_avg.limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC Subsession Hours  
# MAGIC From the Project Ahab report for subsession hours: the average was natural-log transformed to deal with extreme skew with a long right-tail for the purpose of visualization.  

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC Subsession Hours Quantiles

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Correlation between variables

# COMMAND ----------

# MAGIC %md
# MAGIC Limit the outlier values

# COMMAND ----------

lim_uri = 3000
lim_sc = 100
lim_subhrs = 25

# COMMAND ----------

filter_week = ms_1week_avg.filter((F.col("avg_uri") < lim_uri) & (F.col("avg_search_counts") < lim_sc) \
                                 & (F.col("avg_subsession_hours") < lim_subhrs))

# COMMAND ----------

# MAGIC %md
# MAGIC ####URI vs Search Count

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

ms_1week_avg.stat.corr("avg_uri", "avg_search_counts")

# COMMAND ----------

# MAGIC %md
# MAGIC Without the outliers

# COMMAND ----------

display(filter_week)

# COMMAND ----------

filter_week.stat.corr("avg_uri", "avg_search_counts")

# COMMAND ----------

# MAGIC %md
# MAGIC Logs

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC #### URI vs Subsession Hours

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

ms_1week_avg.stat.corr("avg_uri", "avg_subsession_hours")

# COMMAND ----------

# MAGIC %md
# MAGIC Without the outliers

# COMMAND ----------

display(filter_week)

# COMMAND ----------

filter_week.stat.corr("avg_uri", "avg_subsession_hours")

# COMMAND ----------

# MAGIC %md
# MAGIC Logs

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC #### URI vs Active Ticks

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

ms_1week_avg.stat.corr("avg_uri", "avg_active_tick_hrs")

# COMMAND ----------

# MAGIC %md
# MAGIC Without outliers

# COMMAND ----------

display(filter_week)

# COMMAND ----------

filter_week.stat.corr("avg_uri", "avg_active_tick_hrs")

# COMMAND ----------

# MAGIC %md
# MAGIC Logs

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Cumulative Histogram
# MAGIC Get a sample of values from the week to generate a nice cumulative historgram 

# COMMAND ----------

sample = ms_1week_avg.sample(False, .01)
sample_arr_uri = array(sample.select('avg_uri').rdd.flatMap(lambda x: x).collect())
sample_arr_uri

# COMMAND ----------

arr_uri = array(ms_1week_avg.select('avg_uri').rdd.flatMap(lambda x: x).collect())
arr_uri

# COMMAND ----------

week_quantiles = ms_1week_avg.stat.approxQuantile("avg_uri", prob, relError)
week_quantiles

# COMMAND ----------

sample_quantiles = sample.stat.approxQuantile("avg_uri", prob, relError)
sample_quantiles

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the quantiles above for the full data and the sample, the sample is pretty representative of the whole

# COMMAND ----------

n_bins = 500
plt.gcf().clear()

fig, ax = plt.subplots(figsize=(8, 5))

# plot the cumulative histogram
n, bins, patches = ax.hist(sample_arr_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Label')
#n, bins, patches = ax.hist(arr_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Label')

# tidy up the figure
ax.set_xlim(-10,600)

ax.grid(b=True)
ax.set_title('Cumulative step histogram')
ax.set_xlabel('Average URI')
ax.set_ylabel('Likelihood of occurrence')
#ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.axhline(y=.95, linewidth=1, color='r', linestyle="--")
ax.axhline(y=.80, linewidth=1, color='r', linestyle="--")

# COMMAND ----------

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Frequency Table of unique values of URI totals

# COMMAND ----------

sqlContext.registerDataFrameAsTable(ms_1week_avg,"ms_1week_avg")

uri_freq = (
  spark.sql("""
    SELECT 
      avg_uri, 
      count(distinct(client_id)) AS num_clients
    FROM 
      ms_1week_avg
    GROUP BY avg_uri
    ORDER BY avg_uri
    """)
)


# COMMAND ----------

uri_freq.show()

# COMMAND ----------

uri_freq.count()

# COMMAND ----------

uri_freq.describe().show()

# COMMAND ----------

high_uri_high_freq = uri_freq.filter((F.col("num_clients") > 1) & (F.col("avg_uri") > 2000))
high_uri_high_freq.count()

# COMMAND ----------

veryhigh_uri_high_freq = uri_freq.filter((F.col("num_clients") > 1) & (F.col("avg_uri") > 3000))
veryhigh_uri_high_freq.count()

# COMMAND ----------

# MAGIC %md
# MAGIC There are 118 cases of URI over 2000 that have more than one client.  There are 27 cases of URI over 3000 that have more than one client.  
# MAGIC There are 4 clients with a URI of 2128.  Why?  Can we tell anything more about these 4 clients?  
# MAGIC There are 3 clients with a URI of 3149 and 3 clients with a URI of 3640.

# COMMAND ----------

display(high_uri_high_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC TODO Cross check by comparing to other weeks, and other samples

# COMMAND ----------

# MAGIC %md
# MAGIC <a href="$./Heavy User Analysis Outliers and Oddities">Link to next notebook of Heavy User EDA</a>

# COMMAND ----------


