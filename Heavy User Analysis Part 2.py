# Databricks notebook source
# MAGIC %md
# MAGIC # Analysis of Heavy Users - Part 2

# COMMAND ----------

# MAGIC %md
# MAGIC <a href="$./Heavy User Analysis">Link to previous notebook of Heavy User EDA</a>

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
# MAGIC ###Number of days per week over a couple of weeks

# COMMAND ----------

# Get all the data from main_summary for 4 weeks
ms_4week_sum = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum((coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600) AS td_active_tick_hrs,
      sum(subsession_length/3600) AS td_subsession_hours
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    GROUP BY
        1, 2
    """.format(week_1_start, week_4_end, sample_id)
    )

search_4wk = spark.sql("""
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
    """.format(week_1_start, week_4_end, sample_id)
    )


# COMMAND ----------

search_4wk = search_4wk.na.fill(0)
# Sum over the day
search_4wk_sum = search_4wk.groupBy("client_id", "submission_date_s3").agg(F.sum(F.col("sap")+F.col("in_content_organic")))  \
                        .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts')
# Join the daily search counts with the other daily counts 
ms_4week = ms_4week_raw.join(search_4wk_sum, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)
display(ms_4week)

# COMMAND ----------

# Get the number of the week and add a column for that week's number

PERIODS = {}
N_WEEKS = 4
for i in range(1, N_WEEKS + 1):
    PERIODS[i] = {
        'start': (i - 1) * 7,
        'end': (i - 1) * 7 + 6
    }

udf = F.udf

def date_diff(d1, fmt='%Y%m%d'):
    """
    Returns days elapsed from week_1_start to d1 as an integer
    
    Params:
    d1 (str)
    fmt (str): format of d1 and d2 (must be the same)
    
    >>> date_diff('20170205', '20170201')
    4
    
    >>> date_diff('20170201', '20170205)
    -4
    """
    try:
        return (pd.to_datetime(d1, format=fmt) - 
                pd.to_datetime(week_1_start, format=fmt)).days
    except:
        return None
    

@udf(returnType=st.IntegerType())
def get_period(submission_date_s3):
    """
    Given a submission_date_s3,
    returns what period a ping belongs to. This 
    is a spark UDF.
    
    Params:
    submission_date_s3 (col): a ping's submission_date to s3
    
    Global:
    PERIODS (dict): defined globally based on n-week method
    
    Returns an integer indicating the retention period
    """
    if week_1_start is not None:
        diff = date_diff(submission_date_s3)
        if diff >= 0: 
            for period in sorted(PERIODS):
                if diff <= PERIODS[period]['end']:
                    return period


# COMMAND ----------

def add_week(sdf):

  # Determine which week the date is in 
  weeks=  (
    sdf
    .withColumn("week", get_period("submission_date_s3"))
  )
  
  return weeks

ms_4week_wc = add_week(ms_4week)
display(ms_4week_wc)

# COMMAND ----------

# MAGIC %md
# MAGIC Eliminate any rows with all zero counts

# COMMAND ----------

all_zero = ms_4week_wc.filter((F.col("td_uri") == 0) & (F.col("td_search_counts") == 0)  \
                               & (F.col("td_subsession_hours") == 0) & (F.col("td_active_tick_hrs") == 0))
all_zero.count()

# COMMAND ----------

ms_4week_nz = ms_4week_wc.filter((F.col("td_uri") > 0) | (F.col("td_search_counts") > 0)  \
                               | (F.col("td_subsession_hours") > 0) | (F.col("td_active_tick_hrs") > 0))

# COMMAND ----------

display(ms_4week_nz)

# COMMAND ----------

# Count all the days for a client for a given week
ms_4week_count = ms_4week_nz.groupby(['client_id', 'week']).count().withColumnRenamed('count','num_days').sort('client_id', 'week')
display(ms_4week_count)

# COMMAND ----------

# MAGIC %md
# MAGIC At least 1 non-zero count  
# MAGIC This shows how many clients used Firefox for a number of days of the week during 4 different weeks.  For example, approximately 500,000 clients had at least 1 non-zero count value only 1 day a week and that was consistent across all 4 weeks from mid September to mid October.

# COMMAND ----------

any_week_freq = ms_4week_count.groupby("num_days").pivot("week").agg(F.countDistinct("client_id")).sort("num_days")
display(any_week_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC How much variance is there in the number of days a week a client has at least 1 non-zero count?

# COMMAND ----------

client_varpop = ms_4week_count.groupby("client_id").agg(F.var_pop("num_days").alias("variance_num_days")).sort("client_id")
display(client_varpop)

# COMMAND ----------

client_varpop.describe('variance_num_days').show()

# COMMAND ----------

# MAGIC %md
# MAGIC Get the Active Daily Users 

# COMMAND ----------

DAU_4week = ms_4week_wc.filter((F.col("td_uri") >= 5))
DAU_4week.count()

# COMMAND ----------

# Count all the days for a client for a given week
DAU_4week_count = DAU_4week.groupby(['client_id', 'week']).count().withColumnRenamed('count','num_days')
display(DAU_4week_count)

# COMMAND ----------

# MAGIC %md
# MAGIC aDAU  
# MAGIC This shows how many clients used Firefox for a number of days of the week during 4 different weeks.  For example, approximately 380,000 clients had 5 or more URIs only 1 day a week and that was consistent across all 4 weeks from mid September to mid October.

# COMMAND ----------

DAU_week_freq = DAU_4week_count.groupby("num_days").pivot("week").agg(F.countDistinct("client_id")).sort("num_days")
display(DAU_week_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks fairly similar to the graph above for any of the 4 values being non-zero.
# MAGIC Compare normalized DAU week1 with normalized any non-zero week1

# COMMAND ----------

DAU_week1_freq = DAU_week_freq.withColumnRenamed("1", "DAUweek1")
any_week1_freq = any_week_freq.withColumnRenamed("1", "anyweek1")
week1_freq = DAU_week1_freq.alias('a').join(any_week1_freq.alias('b'),F.col('b.num_days') == F.col('a.num_days')) \
         .select([F.col('a.num_days'),F.col('a.DAUweek1')] + [F.col('b.anyweek1')]).sort("num_days")

# COMMAND ----------

display(week1_freq)

# COMMAND ----------

pd_week1 = week1_freq.toPandas()

# COMMAND ----------

maxDAU = pd_week1['DAUweek1'].max()
minDAU = pd_week1['DAUweek1'].min()
maxany = pd_week1['anyweek1'].max()
minany = pd_week1['anyweek1'].min()

# COMMAND ----------

pd_week1.set_index('num_days', inplace=True)
pd_week1['normDAU'] = (pd_week1['DAUweek1'] - minDAU)/(maxDAU - minDAU)
pd_week1['normany'] = (pd_week1['anyweek1'] - minany)/(maxany - minany)

# COMMAND ----------

pd_week1.head(7)

# COMMAND ----------

plt.clf()

pd_week1.plot(y=['normany', 'normDAU'], kind='bar', figsize=(7,3), rot=0, width=.8)

plt.xlabel('Number of Days')
plt.ylabel('Normalized count of clients')

display()

# COMMAND ----------

# MAGIC %md
# MAGIC Clients with 75 or more URIs a day for a number of days of the week during 4 different weeks.

# COMMAND ----------

uri75_4week = ms_4week_wc.filter((F.col("td_uri") >= 75))
# Count all the days for a client for a given week
uri75_4week_count = uri75_4week.groupby(['client_id', 'week']).count().withColumnRenamed('count','num_days')
uri75_week_freq = uri75_4week_count.groupby("num_days").pivot("week").agg(F.countDistinct("client_id")).sort("num_days")
display(uri75_week_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC Clients with 100 or more URIs a day for a number of days of the week during 4 different weeks.

# COMMAND ----------

uri100_4week = ms_4week_wc.filter((F.col("td_uri") >= 100))
# Count all the days for a client for a given week
uri100_4week_count = uri100_4week.groupby(['client_id', 'week']).count().withColumnRenamed('count','num_days')
uri100_week_freq = uri100_4week_count.groupby("num_days").pivot("week").agg(F.countDistinct("client_id")).sort("num_days")
display(uri100_week_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC Clients with 300 or more URIs a day for a number of days of the week during 4 different weeks.

# COMMAND ----------

uri300_4week = ms_4week_wc.filter((F.col("td_uri") >= 300))
# Count all the days for a client for a given week
uri300_4week_count = uri300_4week.groupby(['client_id', 'week']).count().withColumnRenamed('count','num_days')
uri300_week_freq = uri300_4week_count.groupby("num_days").pivot("week").agg(F.countDistinct("client_id")).sort("num_days")
display(uri300_week_freq)

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the percent of daily active users who had 75, 100, and 300 or more URIs a day for one week.

# COMMAND ----------

uri75_week1_freq = uri75_week_freq.withColumnRenamed("1", "uri75week1")
uri100_week1_freq = uri100_week_freq.withColumnRenamed("1", "uri100week1")
uri300_week1_freq = uri300_week_freq.withColumnRenamed("1", "uri300week1")

# COMMAND ----------

pct75_week1_freq = DAU_week1_freq.alias('a').join(uri75_week1_freq.alias('b'),F.col('b.num_days') == F.col('a.num_days')) \
         .select([F.col('a.num_days'),F.col('a.DAUweek1')] + [F.col('b.uri75week1')]).sort("num_days")

# COMMAND ----------

pct100_week1_freq = uri100_week1_freq.alias('a').join(uri300_week1_freq.alias('b'),F.col('b.num_days') == F.col('a.num_days')) \
         .select([F.col('a.num_days'),F.col('a.uri100week1')] + [F.col('b.uri300week1')]).sort("num_days")

# COMMAND ----------

pct_week1_freq = pct75_week1_freq.join(pct100_week1_freq, "num_days").sort("num_days")
display(pct_week1_freq)

# COMMAND ----------

# Calculate percentages
pct_week1_freq = pct_week1_freq.withColumn('pct75', F.col('uri75week1') / F.col('DAUweek1') *100)
pct_week1_freq = pct_week1_freq.withColumn('pct100', F.col('uri100week1') / F.col('DAUweek1') * 100)                
pct_week1_freq = pct_week1_freq.withColumn('pct300', F.col('uri300week1') / F.col('DAUweek1') * 100)       

# COMMAND ----------

# MAGIC %md
# MAGIC Percent of Active Daily Users who have more than 75, 100 or 300 URIs a day for a given number of days a week.

# COMMAND ----------

display(pct_week1_freq)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Look at addon counts, active addons, tab count, window count and unique domains based on the aggregate method used for the clients_daily table.

# COMMAND ----------

ms_1week_raw = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum((coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600) AS td_active_tick_hrs,
      sum(subsession_length/3600) AS td_subsession_hours,
      mean(active_addons_count) as mean_active_addons,
      max(scalar_parent_browser_engagement_max_concurrent_tab_count) as max_tab_count,
      max(scalar_parent_browser_engagement_max_concurrent_window_count) as max_window_count,
      mean(scalar_parent_browser_engagement_unique_domains_count) as mean_domains_count
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

ms_1week_raw = ms_1week_raw.na.fill(0)
display(ms_1week_raw)

# COMMAND ----------

ms_1week_7_avg = ms_1week_raw.groupBy('client_id').avg() \
  .withColumnRenamed('avg(td_uri)','avg_uri') \
  .withColumnRenamed('avg(td_active_tick_hrs)', 'avg_active_tick_hrs') \
  .withColumnRenamed('avg(td_subsession_hours)', 'avg_subsession_hours') \
  .withColumnRenamed('avg(mean_active_addons)', 'avg_addons') \
  .withColumnRenamed('avg(max_tab_count)', 'avg_tab_count') \
  .withColumnRenamed('avg(max_window_count)', 'avg_window_count') \
  .withColumnRenamed('avg(mean_domains_count)', 'avg_domains_count')

# COMMAND ----------

display(ms_1week_7_avg)

# COMMAND ----------

search_wk = search_wk.na.fill(0)
# Sum over the day
search_wk_sum = search_wk.groupBy("client_id", "submission_date_s3").agg(F.sum(F.col("sap")+F.col("in_content_organic")))  \
                        .withColumnRenamed('sum((sap + in_content_organic))','sum_search_counts')
# Average over the week
search_wk_avg = search_wk_sum.groupby('client_id').avg() \
                                 .withColumnRenamed('avg(sum_search_counts)','avg_search_counts')

# COMMAND ----------

ms_1week_avg = ms_1week_7_avg.join(search_wk_avg, ['client_id'], 'full_outer').na.fill(0)

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

ms_1week_avg.describe().show()

# COMMAND ----------



# COMMAND ----------

addons_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_addons", prob, relError)
addons_week_quantiles

# COMMAND ----------

tab_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_tab_count", prob, relError)
tab_week_quantiles

# COMMAND ----------

window_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_window_count", prob, relError)
window_week_quantiles

# COMMAND ----------

domains_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_domains_count", prob, relError)
domains_week_quantiles

# COMMAND ----------

# Histogram of addons
display(ms_1week_avg)

# COMMAND ----------

# Quantile of addons
display(ms_1week_avg)

# COMMAND ----------

# Histogram of tab count
display(ms_1week_avg)

# COMMAND ----------

# Quantile of tab count
display(ms_1week_avg)

# COMMAND ----------

# Histogram of window count
display(ms_1week_avg)

# COMMAND ----------

# Quantile of window count
display(ms_1week_avg)

# COMMAND ----------

# Histogram of domains count
display(ms_1week_avg)

# COMMAND ----------

# Quantile of domains count
display(ms_1week_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Correlation between variables

# COMMAND ----------

# Limit the outliers
lim_uri = 3000
lim_addons = 500
lim_tab = 15
lim_window = 4
lim_domain = 15

# COMMAND ----------

filter_week = ms_1week_avg.filter((F.col("avg_uri") < lim_uri) \
                                  & (F.col("avg_addons") < lim_addons) \
                                  & (F.col("avg_tab_count") < lim_tab) \
                                  & (F.col("avg_window_count") < lim_window) 
                                  & (F.col("avg_domains_count") < lim_domain))

# COMMAND ----------

# MAGIC %md
# MAGIC ####URI vs Add ons

# COMMAND ----------

# Scatter plot 
display(ms_1week_avg)

# COMMAND ----------

display(filter_week)

# COMMAND ----------

# MAGIC %md
# MAGIC ####URI vs Tab Count

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

display(filter_week)

# COMMAND ----------

# MAGIC %md
# MAGIC ####URI vs Window Count

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

display(filter_week)

# COMMAND ----------

# MAGIC %md
# MAGIC ####URI vs Domains Count

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC TODO There are 118 cases of URI over 2000 that have more than one client. There are 27 cases of URI over 3000 that have more than one client.

# COMMAND ----------

high_uri = ms_1week_avg.filter(F.col("avg_uri") > 2000).sort("avg_uri", "client_id")

# COMMAND ----------

display(high_uri)

# COMMAND ----------

high_uri.count()

# COMMAND ----------

uri2128 = ms_1week_avg.filter(F.col("avg_uri") == 2128)

# COMMAND ----------



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

display(uri_freq)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

cd_1week = spark.sql("""
    SELECT *
    FROM clients_daily
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    """.format(week_1_start, week_1_end, sample_id)
    )

# COMMAND ----------

display(cd_1week)

# COMMAND ----------

# MAGIC %md
# MAGIC What is going on in the records with outlier values for uri, active_ticks, subsession_hours and search_counts?  
# MAGIC Look at other values in main_summary like ...

# COMMAND ----------



# COMMAND ----------


