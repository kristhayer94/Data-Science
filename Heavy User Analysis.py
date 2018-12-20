# Databricks notebook source
# MAGIC %md
# MAGIC # Definition of Heavy Users
# MAGIC Determine a cutoff for "heavy"

# COMMAND ----------

# MAGIC %md
# MAGIC Task: Use summary stats of a sample of data.  Do some visualizations, look at percentiles.  
# MAGIC Use one sample_id for a week of submission_date_s3's in main_summary.

# COMMAND ----------

# MAGIC %md
# MAGIC Based on analysis done by Brendan Colloran from the Strategy and Insights team in 2016, Saptarshi Guha in 2017 and Project Ahab in 2018, I will be looking at URI count, search count, subsession hours and active ticks.

# COMMAND ----------

ms_1day = (
  spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum(active_ticks*5/3600) AS td_active_ticks,
      sum(subsession_length/3600) AS td_subsession_hours
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 = '20180923'
      AND sample_id = '42'
    GROUP BY
        1, 2
  """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Active ticks are in 5 second increments, so it is converted to hours.  
# MAGIC Subsession length is in seconds, so it is converted to hours.  
# MAGIC   
# MAGIC **Used 1 day first, then went back and added an average over a week.**

# COMMAND ----------

ms_1day.show()

# COMMAND ----------

ms_1week = (
  spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum(active_ticks*5/3600) AS td_active_ticks,
      sum(subsession_length/3600) AS td_subsession_hours
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '20180923'
      AND submission_date_s3 <= '20180929'
      AND sample_id = '42'
    GROUP BY
        1, 2
  """)
)

# COMMAND ----------

ms_1week.show()

# COMMAND ----------

ms_1week.count()

# COMMAND ----------

# MAGIC %md
# MAGIC **This would be the place to cap the erroneous large data before averaging.  
# MAGIC Cap the subsession hours to 25.  Any other caps?**

# COMMAND ----------

from pyspark.sql.functions import when

# Replace subsession hours > 25 with 25
ms_1week_cap = ms_1week.withColumn("td_subsession_hours", \
              when(ms_1week["td_subsession_hours"] > 25, 25).otherwise(ms_1week["td_subsession_hours"]))

# COMMAND ----------

ms_1week_cap.show()

# COMMAND ----------

ms_1week.describe("td_subsession_hours").show()

# COMMAND ----------

ms_1week_cap.describe("td_subsession_hours").show()

# COMMAND ----------

ms_1week_avg = ms_1week_cap.groupby('client_id').avg()

# COMMAND ----------

ms_1week_avg.show()

# COMMAND ----------

# Rename columns
ms_1week_avg = (ms_1week_avg
   .withColumnRenamed('avg(td_uri)','avg_uri')
   .withColumnRenamed('avg(td_active_ticks)', 'avg_active_ticks')
   .withColumnRenamed('avg(td_subsession_hours)', 'avg_subsession_hours'))

# COMMAND ----------

ms_1week_avg.show()

# COMMAND ----------

# Number of rows of client_ids
ms_1week_avg.count()

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Number of distinct client_ids
ms_1week_avg.agg(countDistinct("client_id")).show()

# COMMAND ----------

# Number of rows of client_ids
ms_1day.count()

# COMMAND ----------

# Number of distinct client_ids
ms_1day.agg(countDistinct("client_id")).show()

# COMMAND ----------

ms_1day_sc = (
  spark.sql("""
  SELECT 
    client_id,
    submission_date_s3,
    search_counts
  FROM main_summary
  WHERE 
    app_name='Firefox'
    AND submission_date_s3 = '20180923'
    AND sample_id = '42'
  """)
)

# COMMAND ----------

ms_1day_sc.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Extract the search counts from the array in search_counts column for 1 day

# COMMAND ----------

import pyspark.sql.functions as F

# explode to get structure
ms_1day_sc=ms_1day_sc.withColumn('exploded', F.explode('search_counts'))
# get the count in a separate column
ms_1day_sc=ms_1day_sc.withColumn('search_count', F.col('exploded').getItem("count"))
ms_1day_sc=ms_1day_sc.drop('search_counts', 'exploded')
ms_1day_sc.show()

# COMMAND ----------

ms_1day_sc_g = ms_1day_sc.groupby('client_id').sum()
ms_1day_sc_g = (ms_1day_sc_g
   .withColumnRenamed('sum(search_count)','td_search_counts'))
ms_1day_sc_g.show()

# COMMAND ----------

#Number of rows of client_ids
ms_1day_sc_g.count()

# COMMAND ----------

#Number of distinct client_ids
ms_1day_sc_g.agg(countDistinct("client_id")).show()

# COMMAND ----------

ms_1day_ac = ms_1day.join(ms_1day_sc_g, ['client_id'], 'full_outer').na.fill(0)

# COMMAND ----------

ms_1day_ac.show()

# COMMAND ----------

#Number of rows of client_ids
ms_1day_ac.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Extract the search counts from the array in search_counts column for the week

# COMMAND ----------

ms_1week_sc = (
  spark.sql("""
  SELECT 
    client_id,
    submission_date_s3,
    search_counts
  FROM main_summary
  WHERE 
    app_name='Firefox'
    AND submission_date_s3 >= '20180923'
    AND submission_date_s3 <= '20180929'
    AND sample_id = '42'
  """)
)

# COMMAND ----------

ms_1week_sc.show()

# COMMAND ----------

# explode to get structure
ms_1week_sc=ms_1week_sc.withColumn('exploded', F.explode('search_counts'))
# get the count in a separate column
ms_1week_sc=ms_1week_sc.withColumn('search_count', F.col('exploded').getItem("count"))
ms_1week_sc=ms_1week_sc.drop('search_counts', 'exploded')
ms_1week_sc.show()

# COMMAND ----------

ms_1week_sc_avg = ms_1week_sc.groupby('client_id').avg()

# COMMAND ----------

ms_1week_sc_avg.show()

# COMMAND ----------

# Rename columns
ms_1week_sc_avg = (ms_1week_sc_avg
   .withColumnRenamed('avg(search_count)','avg_search_counts'))

# COMMAND ----------

ms_1week_sc_avg.show()

# COMMAND ----------

ms_1week_ac = ms_1week_avg.join(ms_1week_sc_avg, ['client_id'], 'full_outer').na.fill(0)

# COMMAND ----------

ms_1week_ac.show()

# COMMAND ----------

ms_1week_ac.count()

# COMMAND ----------

ms_1week_ac.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC **There are a lot of very high max values which seems questionable. Is there an explanation for these high values? Should there be a cap on the values?**  
# MAGIC   
# MAGIC From the Project Ahab report: Subsession hours is subject to measurement error due to ping reporting, among other issues. For example, one user recorded 1.9M hours (216 years) in one 24-hour period. This measure should be a maximum of 25 hours-- 25 rather than 24 because there can be up to 1 hour lag in reporting. 

# COMMAND ----------

prob = (0.25,0.5,0.75)
relError = 0

# COMMAND ----------

uri_day_quantiles = ms_1day_ac.stat.approxQuantile("td_uri", prob, relError)
uri_day_quantiles

# COMMAND ----------

uri_week_quantiles = ms_1week_ac.stat.approxQuantile("avg_uri", prob, relError)
uri_week_quantiles

# COMMAND ----------

uri_outliers = ms_1week_ac.filter(ms_1week_ac.avg_uri > uri_week_quantiles[2])
uri_outliers.count()

# COMMAND ----------

uri_outliers.show()

# COMMAND ----------

active_ticks_day_quantiles = ms_1day_ac.stat.approxQuantile("td_active_ticks", prob, relError)
active_ticks_day_quantiles

# COMMAND ----------

active_ticks_week_quantiles = ms_1week_ac.stat.approxQuantile("avg_active_ticks", prob, relError)
active_ticks_week_quantiles                           

# COMMAND ----------

active_ticks_outliers = ms_1week_ac.filter(ms_1week_ac.avg_active_ticks > active_ticks_week_quantiles[2])
active_ticks_outliers.count()

# COMMAND ----------

active_ticks_outliers.show()

# COMMAND ----------

subsession_hours_day_quantiles = ms_1day_ac.stat.approxQuantile("td_subsession_hours", prob, relError)
subsession_hours_day_quantiles

# COMMAND ----------

subsession_hours_week_quantiles = ms_1week_ac.stat.approxQuantile("avg_subsession_hours", prob, relError)
subsession_hours_week_quantiles                           

# COMMAND ----------

subsession_hours_outliers = ms_1week_ac.filter(ms_1week_ac.avg_subsession_hours > subsession_hours_week_quantiles[2])
subsession_hours_outliers.count()

# COMMAND ----------

subsession_hours_outliers.show()

# COMMAND ----------

search_counts_day_quantiles = ms_1day_ac.stat.approxQuantile("td_search_counts", prob, relError)
search_counts_day_quantiles

# COMMAND ----------

search_counts_week_quantiles = ms_1week_ac.stat.approxQuantile("avg_search_counts", prob, relError)
search_counts_week_quantiles       

# COMMAND ----------

search_counts_outliers = ms_1week_ac.filter(ms_1week_ac.avg_search_counts > search_counts_week_quantiles[2])
search_counts_outliers.count()

# COMMAND ----------

search_counts_outliers.show()

# COMMAND ----------

# MAGIC %md
# MAGIC How many rows are outliers in all 4 measurements?

# COMMAND ----------

same_outlier_2 = uri_outliers.join(active_ticks_outliers, ['client_id'], "left_semi")
same_outlier_2.count()

# COMMAND ----------

same_outlier_2.show()

# COMMAND ----------

same_outlier_3 = same_outlier_2.join(subsession_hours_outliers, ['client_id'], "left_semi")
same_outlier_3.count()

# COMMAND ----------

same_outlier_3.show()

# COMMAND ----------

same_outlier_4 = same_outlier_3.join(search_counts_outliers, ['client_id'], "left_semi")
same_outlier_4.count()

# COMMAND ----------

same_outlier_4.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Time to make some visualizations.  
# MAGIC Some example notebooks would be helpful.  
# MAGIC What package to plot in - matplotlib, seaborn?

# COMMAND ----------

# MAGIC %md
# MAGIC I was thinking scatter plots between the 4 columns to see how well they correlate to each other.  
# MAGIC I was thinking histograms of each of the 4 columns.  
# MAGIC **Do I need to transform the data before I plot it?**

# COMMAND ----------

# MAGIC %md
# MAGIC From the Project Ahab report for URI count: The distribution of URI count is extremely skewed with a long right-tail, so the analysis is displayed using the natural log of the weekly average of daily URI count.  

# COMMAND ----------

from pyspark.sql.functions import log1p
ms_1week_logs = ms_1week_ac.withColumn("uri_log", log1p("avg_uri"))
ms_1week_logs = ms_1week_logs.withColumn("active_ticks_log", log1p("avg_active_ticks"))
ms_1week_logs = ms_1week_logs.withColumn("subsession_hours_log", log1p("avg_subsession_hours"))
ms_1week_logs = ms_1week_logs.withColumn("search_counts_log", log1p("avg_search_counts"))

# COMMAND ----------

ms_1week_logs.show()

# COMMAND ----------

ms_1week_logs.describe().show()

# COMMAND ----------

# URI distribution, agg = count, 50 bins
display(ms_1week_ac.limit(1000))

# COMMAND ----------

# URI distribution log transformed, agg = count, 50 bins
display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC The Quantiles chart is based on the first 1000 rows, so it may not be accurate.  After checking, the values are quite close to the approxQuantile 75% values

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------

# MAGIC %md
# MAGIC From the Project Ahab report for search count: The natural log transformation was applied after averaging.  

# COMMAND ----------

display(ms_1week_ac.limit(1000))

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------

display(ms_1week_ac.limit(1000))

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------

# MAGIC %md
# MAGIC From the Project Ahab report for subsession hours: the average was natural-log transformed to deal with extreme skew with a long right-tail for the purpose of visualization.  

# COMMAND ----------

display(ms_1week_ac.limit(1000))

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC These scatter plots would be more useful without the outliers.

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------

from pyspark.sql.functions import col

filterSC = ms_1week_ac.filter((col("avg_uri") < 3000))
filterSC = filterDF.filter((col("avg_search_counts") < 100))
display(filterSC)

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

display(ms_1week_ac)

# COMMAND ----------

display(ms_1week_logs)

# COMMAND ----------

# MAGIC %md
# MAGIC Stuff I tried that didn't really do anything meaningful.

# COMMAND ----------

from numpy import array
# Get 1000 values from ms_1week_ac
#uri_lt_3000 = ms_1week_ac.filter(ms_1week_ac["avg_uri"]<3000)
sample = ms_1week_ac.sample(False, .001)
arr_uri = array(sample.select('avg_uri').rdd.flatMap(lambda x: x).collect())
arr_uri

# COMMAND ----------

sample.count()

# COMMAND ----------

week_quantiles = ms_1week_ac.stat.approxQuantile("avg_uri", (0.05, 0.25, 0.5, 0.75, 0.95), 0)
week_quantiles

# COMMAND ----------

sample_quantiles = sample.stat.approxQuantile("avg_uri", (0.05, 0.25, 0.5, 0.75, 0.95), 0)
sample_quantiles

# COMMAND ----------

n_bins = 300

fig, ax = plt.subplots(figsize=(8, 5))

# plot the cumulative histogram
n, bins, patches = ax.hist(arr_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Label')

# tidy up the figure
#ax.grid(b=True, color='k',which='both', axis='both')
#ax.grid(b=True, color='b')
#ax.set_xlim(0,500)

ax.grid(b=True)
ax.set_title('Cumulative step histogram')
ax.set_xlabel('Average URI')
ax.set_ylabel('Likelihood of occurrence')
#ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#ax.axhline(y=.95, linewidth=1, color='r', linestyle="--")

# COMMAND ----------

display(fig)

# COMMAND ----------



# COMMAND ----------

# frequency table of unique values of uri totals.
sqlContext.registerDataFrameAsTable(ms_1week_ac,"ms_1week_ac")

uri_freq = (
  spark.sql("""
    SELECT 
      avg_uri, 
      count(distinct(client_id)) AS num_clients
    FROM 
      ms_1week_ac
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


