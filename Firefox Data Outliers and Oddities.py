# Databricks notebook source
import pyspark.sql.functions as F
import pyspark.sql.types as st
import numpy as np
from matplotlib.patches import Polygon
from sklearn import preprocessing
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ###Prepare Data

# COMMAND ----------

#Global variables
sample_id = 42
week_1_start = '20180923'
week_1_end = '20180929'
week_4_end = '20181020'
week_12_end = '20181215'
week_16_end = '20190112'
day_1_date = '20180925'
day_1_old = '20180125'
prob = (0.05, 0.25, 0.5, 0.75, 0.95)
ninetyfive_ind = 4 #index for 95% quantile
relError = 0
outlier_prob = (0.95, 0.96, 0.97, 0.98, 0.99, 0.995)

# COMMAND ----------

ms_1week_raw = spark.sql("""
    SELECT
        client_id,
        coalesce(scalar_parent_browser_engagement_total_uri_count, 0) AS uri_count,
        coalesce(scalar_parent_browser_engagement_active_ticks, 0) AS active_ticks,
        (coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600 AS active_tick_hrs,        
        subsession_length,
        (subsession_length/3600) AS subsession_hours,
        session_length,
        profile_subsession_counter,
        submission_date_s3,
        session_start_date,
        subsession_start_date,
        reason,
        active_addons_count,
        scalar_parent_browser_engagement_max_concurrent_tab_count AS tab_count,
        scalar_parent_browser_engagement_max_concurrent_window_count AS window_count,
        scalar_parent_browser_engagement_unique_domains_count AS domains_count,
        profile_creation_date,
        profile_reset_date,
        previous_build_id,
        normalized_channel,
        os,
        normalized_os_version,
        windows_build_number,
        install_year,
        creation_date,
        distribution_id,
        submission_date,
        app_build_id,
        app_display_version,
        update_channel,
        update_enabled,
        update_auto_download,
        active_experiment_branch,
        timezone_offset,
        vendor,
        is_default_browser,
        default_search_engine,
        client_submission_date,
        places_bookmarks_count,
        places_pages_count,
        scalar_content_telemetry_event_counts AS telem_event_counts,
        scalar_parent_browser_engagement_tab_open_event_count AS tab_event_count,
        scalar_parent_browser_engagement_window_open_event_count AS window_event_count,
        scalar_parent_browser_errors_collected_count AS errors_collected_count,
        scalar_parent_browser_feeds_livebookmark_count AS livebookmark_count,
        scalar_parent_devtools_current_theme AS current_theme,
        scalar_parent_formautofill_availability AS formautofill_availability,
        scalar_parent_media_page_count AS media_page_count, 
        country,
        city,
        geo_subdivision1,
        locale,
        antivirus,
        antispyware,
        firewall,
        session_id,
        subsession_id,
        sync_configured,
        sync_count_desktop,
        sync_count_mobile,
        disabled_addons_ids,
        active_theme,
        user_prefs,
        experiments,
        sample_id 
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    ORDER BY
        client_id,
        submission_date_s3,
        profile_subsession_counter
    """.format(week_1_start, week_1_end, sample_id)
    )

# From telemetry docs for how clients_daily deteremines values
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

ms_1day_raw = spark.sql("""
    SELECT
        client_id,
        coalesce(scalar_parent_browser_engagement_total_uri_count, 0) AS uri_count,
        coalesce(scalar_parent_browser_engagement_active_ticks, 0) AS active_ticks,
        (coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600 AS active_tick_hrs,        
        subsession_length,
        (subsession_length/3600) AS subsession_hours,
        session_length,
        profile_subsession_counter,
        submission_date_s3,
        session_start_date,
        subsession_start_date,
        subsession_counter,
        reason,
        active_addons_count,
        scalar_parent_browser_engagement_max_concurrent_tab_count AS tab_count,
        scalar_parent_browser_engagement_max_concurrent_window_count AS window_count,
        scalar_parent_browser_engagement_unique_domains_count AS domains_count,
        profile_creation_date,
        profile_reset_date,
        previous_build_id,
        normalized_channel,
        os,
        normalized_os_version,
        windows_build_number,
        install_year,
        creation_date,
        distribution_id,
        submission_date,
        app_build_id,
        app_display_version,
        update_channel,
        update_enabled,
        update_auto_download,
        active_experiment_branch,
        timezone_offset,
        vendor,
        is_default_browser,
        default_search_engine,
        client_submission_date,
        places_bookmarks_count,
        places_pages_count,
        scalar_content_telemetry_event_counts AS telem_event_counts,
        scalar_parent_browser_engagement_tab_open_event_count AS tab_event_count,
        scalar_parent_browser_engagement_window_open_event_count AS window_event_count,
        scalar_parent_browser_errors_collected_count AS errors_collected_count,
        scalar_parent_browser_feeds_livebookmark_count AS livebookmark_count,
        scalar_parent_devtools_current_theme AS current_theme,
        scalar_parent_formautofill_availability AS formautofill_availability,
        scalar_parent_media_page_count AS media_page_count, 
        country,
        city,
        geo_subdivision1,
        locale,
        antivirus,
        antispyware,
        firewall,
        session_id,
        subsession_id,
        sync_configured,
        sync_count_desktop,
        sync_count_mobile,
        disabled_addons_ids,
        active_theme,
        user_prefs,
        experiments,
        sample_id 
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 == '{}'
      AND sample_id = '{}'
    ORDER BY
        client_id,
        profile_subsession_counter
    """.format(day_1_date, sample_id)
    )

ms_1day_sum = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)) AS td_active_ticks,
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)*5/3600) AS td_active_tick_hrs,
      sum(subsession_length/3600) AS td_subsession_hours
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 == '{}'
      AND sample_id = '{}'
    GROUP BY
        1, 2
    """.format(day_1_date, sample_id)
    )

search_day = spark.sql("""
  SELECT client_id,
         submission_date_s3,
         engine,
         SUM(sap) as sap,
         SUM(tagged_sap) as tagged_sap,
         SUM(tagged_follow_on) as tagged_follow_on,
         SUM(organic) as in_content_organic
  FROM search_clients_daily
  WHERE
      submission_date_s3 == '{}'
      AND sample_id = '{}'
  GROUP BY
      1, 2, 3
    """.format(day_1_date, sample_id)
    )

ms_4week_raw = spark.sql("""
    SELECT
        client_id,
        coalesce(scalar_parent_browser_engagement_total_uri_count, 0) AS uri_count,
        coalesce(scalar_parent_browser_engagement_active_ticks, 0) AS active_ticks,
        (coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600 AS active_tick_hrs,        
        subsession_length,
        (subsession_length/3600) AS subsession_hours,
        session_length,
        profile_subsession_counter,
        submission_date_s3,
        session_start_date,
        subsession_start_date,
        reason,
        active_addons_count,
        scalar_parent_browser_engagement_max_concurrent_tab_count AS tab_count,
        scalar_parent_browser_engagement_max_concurrent_window_count AS window_count,
        scalar_parent_browser_engagement_unique_domains_count AS domains_count,
        profile_creation_date,
        profile_reset_date,
        previous_build_id,
        normalized_channel,
        os,
        normalized_os_version,
        windows_build_number,
        install_year,
        creation_date,
        distribution_id,
        submission_date,
        app_build_id,
        app_display_version,
        update_channel,
        update_enabled,
        update_auto_download,
        active_experiment_branch,
        timezone_offset,
        vendor,
        is_default_browser,
        default_search_engine,
        client_submission_date,
        places_bookmarks_count,
        places_pages_count,
        scalar_content_telemetry_event_counts AS telem_event_counts,
        scalar_parent_browser_engagement_tab_open_event_count AS tab_event_count,
        scalar_parent_browser_engagement_window_open_event_count AS window_event_count,
        scalar_parent_browser_errors_collected_count AS errors_collected_count,
        scalar_parent_browser_feeds_livebookmark_count AS livebookmark_count,
        scalar_parent_devtools_current_theme AS current_theme,
        scalar_parent_formautofill_availability AS formautofill_availability,
        scalar_parent_media_page_count AS media_page_count, 
        country,
        city,
        geo_subdivision1,
        locale,
        antivirus,
        antispyware,
        firewall,
        session_id,
        subsession_id,
        sync_configured,
        sync_count_desktop,
        sync_count_mobile,
        disabled_addons_ids,
        active_theme,
        user_prefs,
        experiments,
        sample_id 
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    ORDER BY
        client_id,
        submission_date_s3,
        profile_subsession_counter
    """.format(week_1_start, week_4_end, sample_id)
    )


# COMMAND ----------

# Fill null search counts with 0s
search_wk = search_wk.na.fill(0)
# Sum over the day
search_wk_sum = search_wk.groupBy("client_id", "submission_date_s3").agg(F.sum(F.col("sap")+F.col("in_content_organic")))  \
                        .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts').sort("client_id", "submission_date_s3")

# COMMAND ----------

display(search_wk_sum)

# COMMAND ----------

# Join the search table with the other counts
ms_1week = ms_1week_sum.join(search_wk_sum, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)
display(ms_1week)

# COMMAND ----------

ms_1week_check = ms_1week_sum.where("client_id = '00165152-2a43-4c12-b006-245972dde1d6'")
display(ms_1week_check)

# COMMAND ----------

search_wk_check = search_wk_sum.where("client_id = '00165152-2a43-4c12-b006-245972dde1d6'")
display(search_wk_check)

# COMMAND ----------

ms_1week_check = ms_1week.where("client_id = '00165152-2a43-4c12-b006-245972dde1d6'")
display(ms_1week_check)

# COMMAND ----------

# Average the total daily values over the week
ms_1week_avg = ms_1week.groupBy('client_id').avg() \
  .withColumnRenamed('avg(td_uri)','avg_uri') \
  .withColumnRenamed('avg(td_active_tick_hrs)', 'avg_active_tick_hrs') \
  .withColumnRenamed('avg(td_subsession_hours)', 'avg_subsession_hours') \
  .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')

# COMMAND ----------

# Look at the data for 1 week averaged
display(ms_1week_avg)

# COMMAND ----------

# Look at the summary stats for 1 week averaged
display(ms_1week_avg.describe())

# COMMAND ----------

ms_1week_avg_check = ms_1week_avg.where("client_id = '00165152-2a43-4c12-b006-245972dde1d6'")
display(ms_1week_avg_check)

# COMMAND ----------

# Look at the summary stats for 1 week summed for each day
display(ms_1week.describe())

# COMMAND ----------

# Look at 1 week summed for each day only for uri >= 5
ms_1week_aDAU = ms_1week.where("td_uri >= 5")
display(ms_1week_aDAU)

# COMMAND ----------

# Look at summary stats for 1 week summed for each day for uri >= 5
display(ms_1week_aDAU.describe())

# COMMAND ----------

# Average the total daily values for aDAU over the week - this will not average in all the 0 uri rows 
ms_1week_aDAU_avg = ms_1week_aDAU.groupBy('client_id').avg() \
  .withColumnRenamed('avg(td_uri)','avg_uri') \
  .withColumnRenamed('avg(td_active_tick_hrs)', 'avg_active_tick_hrs') \
  .withColumnRenamed('avg(td_subsession_hours)', 'avg_subsession_hours') \
  .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')

# COMMAND ----------

# Look at the summary stats for aDAU averaged over a week
display(ms_1week_aDAU_avg.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Look at outliers for both a single day and averaged over a week for all records and for aDAU

# COMMAND ----------

# Get 1 day of summed values from the week
ms_1day = ms_1week.where("submission_date_s3 = '20180925'")
display(ms_1day)

# COMMAND ----------

# Look at summary stats for the day
display(ms_1day.describe())

# COMMAND ----------

# How similar is Tuesday to Wednesday?
ms_wday = ms_1week.where("submission_date_s3 = '20180926'")
display(ms_wday.describe())

# COMMAND ----------

# How similar is Wednesday to Thursday?
ms_thday = ms_1week.where("submission_date_s3 = '20180927'")
display(ms_thday.describe())

# COMMAND ----------

# How similar is average over week to individual day?
display(ms_1week_avg.describe())
# stddev's smaller for everything but subsession hours

# COMMAND ----------

ms_1week_avg.stat.corr('avg_uri', 'avg_uri')

# COMMAND ----------

np.corrcoef(tday, wday)

# COMMAND ----------

ms_1week = ms_1week_sum.join(search_wk_sum, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)
display(ms_1week)

# COMMAND ----------

len(tday)

# COMMAND ----------

tday = np.array(ms_1day.rdd.map(lambda p: p.td_uri).collect())
wday = np.array(ms_wday.rdd.map(lambda p: p.td_uri).collect())
thday = np.array(ms_thday.rdd.map(lambda p: p.td_uri).collect())
avg_week = np.array(ms_1week_avg.rdd.map(lambda p: p.avg_uri).collect())

# COMMAND ----------

plt.gcf().clear()
dayDists = ['Tuesday', 'Wednesday', 'Thursday', 'Week Avg']
data = [tday, wday, thday, avg_week]
fig, ax1 = plt.subplots(figsize=(10,6))
bp = ax1.boxplot(data, sym='', whis=[5, 95])
ax1.set_title('URI values')
ax1.set_xlabel('Distribution')
ax1.set_ylabel('Value')
top = 500
bottom = -5
ax1.set_ylim(bottom, top)
ax1.set_xticklabels(dayDists, rotation=0, fontsize=8)
means = [np.mean(x) for x in data]
plt.scatter([1, 2, 3, 4], means)
display(fig)


# COMMAND ----------

# Get 1 day of summed values from the week for aDAU
ms_1day_aDAU = ms_1week_aDAU.where("submission_date_s3 = '20180925'")
display(ms_1day_aDAU)

# COMMAND ----------

# Look at aDAU values for Wednesday
ms_wday_aDAU = ms_1week_aDAU.where("submission_date_s3 = '20180926'")
display(ms_wday_aDAU.describe())

# COMMAND ----------

# Look at aDAU values for Thursday
ms_thday_aDAU = ms_1week_aDAU.where("submission_date_s3 = '20180927'")
display(ms_thday_aDAU.describe())

# COMMAND ----------

# Get arrays of uri values for box plot
tday = np.array(ms_1day.rdd.map(lambda p: p.td_uri).collect())
wday = np.array(ms_wday.rdd.map(lambda p: p.td_uri).collect())
thday = np.array(ms_thday.rdd.map(lambda p: p.td_uri).collect())
avg_week = np.array(ms_1week_avg.rdd.map(lambda p: p.avg_uri).collect())

# COMMAND ----------

# Get arrays of uri values for box plot for aDAU
tday_aDAU = np.array(ms_1day_aDAU.rdd.map(lambda p: p.td_uri).collect())
wday_aDAU = np.array(ms_wday_aDAU.rdd.map(lambda p: p.td_uri).collect())
thday_aDAU = np.array(ms_thday_aDAU.rdd.map(lambda p: p.td_uri).collect())
avg_aDAU_week = np.array(ms_1week_aDAU_avg.rdd.map(lambda p: p.avg_uri).collect())

# COMMAND ----------

plt.gcf().clear()
dayDists = ['Tuesday', 'Tues aDAU', 'Wednesday', 'Wed aDAU', 'Thursday', 'Thurs aDAU', 'Week Avg', 'aDAU Wk Avg']
data = [tday, tday_aDAU, wday, wday_aDAU, thday, thday_aDAU, avg_week, avg_aDAU_week]
fig, ax1 = plt.subplots(figsize=(10,6))
bp = ax1.boxplot(data, sym='', whis=[5, 95])
ax1.set_title('URI values')
ax1.set_xlabel('Distribution')
ax1.set_ylabel('Value')
top = 600
bottom = -5
ax1.set_ylim(bottom, top)
ax1.set_xticklabels(dayDists, rotation=0, fontsize=8)
# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)             
# Hide these grid behind plot objects
ax1.set_axisbelow(True)

means = [np.mean(x) for x in data]
plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], means)
display(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Quantiles

# COMMAND ----------

# URI quantiles for 1 day for all users, 0.05, 0.25, 0.5, 0.75, 0.95
uri_day_quantiles = ms_1day.stat.approxQuantile("td_uri", prob, relError)
uri_day_quantiles

# COMMAND ----------

# Find outliers for 1 day for all users, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995
uri_day_outliers = ms_1day.stat.approxQuantile("td_uri", outlier_prob, relError)
uri_day_outliers

# COMMAND ----------

print "95th to 96th =", uri_day_outliers[1] - uri_day_outliers[0]
print "96th to 97th =", uri_day_outliers[2] - uri_day_outliers[1]
print "97th to 98th =", uri_day_outliers[3] - uri_day_outliers[2]
print "98th to 99th =", uri_day_outliers[4] - uri_day_outliers[3]
print "99th to 99.5th =", uri_day_outliers[5] - uri_day_outliers[4]

# COMMAND ----------

ms_1day.count()

# COMMAND ----------

# How many 1 day records above the 99th percentile?
ms_1day_outliers = ms_1day.where("td_uri > 1071")
display(ms_1day_outliers.describe())

# COMMAND ----------

display(ms_1day_outliers)

# COMMAND ----------

ms_1day_check = ms_1day_raw.where("client_id = '67470b86-ef10-4d5e-beaa-6f7d5cc35f3f'")
display(ms_1day_check)

# COMMAND ----------

sample = ms_1day.sample(False, .00)
sample_arr_uri = array(ms_1day.select('td_uri').rdd.flatMap(lambda x: x).collect())
sample_arr_uri

# COMMAND ----------

n_bins = 1000
plt.gcf().clear()

fig, ax = plt.subplots(figsize=(8, 5))

# plot the cumulative histogram
n, bins, patches = ax.hist(sample_arr_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Why')
#n, bins, patches = ax.hist(arr_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Label')

# tidy up the figure
ax.set_xlim(-10,100)

ax.grid(b=True)
ax.set_title('Cumulative step histogram')
ax.set_xlabel('Total Daily URI')
ax.set_ylabel('Likelihood of occurrence')
#ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.axhline(y=.95, linewidth=1, color='r', linestyle="--")
ax.axhline(y=.80, linewidth=1, color='r', linestyle="--")
display(fig)

# COMMAND ----------

# URI quantiles for 1 week for all users, 0.05, 0.25, 0.5, 0.75, 0.95
uri_week_quantiles = ms_1week_avg.stat.approxQuantile("avg_uri", prob, relError)
uri_week_quantiles

# COMMAND ----------

# Find outliers for 1 week for all users, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995
uri_week_outliers = ms_1week_avg.stat.approxQuantile("avg_uri", outlier_prob, relError)
uri_week_outliers

# COMMAND ----------

print "95th to 96th =", uri_week_outliers[1] - uri_week_outliers[0]
print "96th to 97th =", uri_week_outliers[2] - uri_week_outliers[1]
print "97th to 98th =", uri_week_outliers[3] - uri_week_outliers[2]
print "98th to 99th =", uri_week_outliers[4] - uri_week_outliers[3]
print "99th to 99.5th =", uri_week_outliers[5] - uri_week_outliers[4]

# COMMAND ----------

ms_1week_avg.count()

# COMMAND ----------

# How many 1 week records above the 98th percentile?
ms_1week_outliers = ms_1week_avg.where("avg_uri > 753")
display(ms_1week_outliers.describe())

# COMMAND ----------

display(ms_1week_outliers)

# COMMAND ----------

ms_1week_check = ms_1week_raw.where("client_id = 'd5b70fa4-31df-48a6-863f-c0636c4d3afd'")
display(ms_1week_check)

# COMMAND ----------

ms_1week_check = ms_1week_raw.where("uri_count > 1000 and active_ticks = 0").sort("client_id")
display(ms_1week_check)

# COMMAND ----------

ms_1week_count = ms_1week_check.select("client_id").distinct()
ms_1week_count.count()

# COMMAND ----------

sample = ms_1week_avg.sample(False, .01)
sample_avg_uri = array(sample.select('avg_uri').rdd.flatMap(lambda x: x).collect())
sample_avg_uri

# COMMAND ----------

n_bins = 1200
plt.gcf().clear()

fig, ax = plt.subplots(figsize=(8, 5))

# plot the cumulative histogram
n, bins, patches = ax.hist(sample_avg_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Label')
#n, bins, patches = ax.hist(arr_uri, n_bins, normed=True, histtype='step', cumulative=True, label='Label')

# tidy up the figure
ax.set_xlim(300,800)
ax.set_ylim(.9, 1.05)
ax.grid(b=True)
ax.set_title('Cumulative step histogram')
ax.set_xlabel('Average URI')
ax.set_ylabel('Likelihood of occurrence')
#ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.axhline(y=.98, linewidth=1, color='r', linestyle="--")
ax.axhline(y=.99, linewidth=1, color='r', linestyle="--")
display(fig)

# COMMAND ----------

display(ms_1week_avg)

# COMMAND ----------

# URI quantiles for 1 day for aDAU users
uri_day_aDAU_quantiles = ms_1day_aDAU.stat.approxQuantile("td_uri", prob, relError)
uri_day_aDAU_quantiles

# COMMAND ----------

# Find outliers for 1 day for aDAU users
uri_day_aDAU_outliers = ms_1day_aDAU.stat.approxQuantile("td_uri", outlier_prob, relError)
uri_day_aDAU_outliers

# COMMAND ----------

ms_1day_aDAU.count()

# COMMAND ----------

# How many 1 day records for aDAU above the 98th percentile?
ms_1day_aDAU_outliers = ms_1day_aDAU.where("td_uri > 880")
display(ms_1day_aDAU_outliers.describe())

# COMMAND ----------

display(ms_1day_aDAU)

# COMMAND ----------

# URI quantiles for 1 week for aDAU users
uri_week_aDAU_avg_quantiles = ms_1week_aDAU_avg.stat.approxQuantile("avg_uri", prob, relError)
uri_week_aDAU_avg_quantiles

# COMMAND ----------

# Find outliers for 1 day for aDAU users
uri_week_aDAU_outliers = ms_1week_aDAU_avg.stat.approxQuantile("avg_uri", outlier_prob, relError)
uri_week_aDAU_outliers

# COMMAND ----------

ms_1week_aDAU_avg.count()

# COMMAND ----------

# How many 1 week aDAU records above the 98th percentile?
ms_1week_aDAU_outliers = ms_1week_aDAU_avg.where("avg_uri > 633")
display(ms_1week_aDAU_outliers.describe())

# COMMAND ----------

display(ms_1week_aDAU_avg)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# How many clients with multiple daily pings on same day?
dailyping_day = ms_1day_raw.filter((F.col("reason") == 'daily'))
multping_day = dailyping_day.groupBy("client_id") \
    .count()\
    .where(F.col('count') > 1)
display(multping_day)

# COMMAND ----------

# Summary stats for the multiple daily ping client_ids
display(multping_day.describe())

# COMMAND ----------

# Look at one of the clients
multping_c1 = dailyping_day.where("client_id ='92004b2f-e045-48a9-a20b-87ae2534eb75'").sort("profile_subsession_counter")
display(multping_c1)

# COMMAND ----------

# Look at summary stats for 1 client with multiple daily pings
display(multping_c1.describe())

# COMMAND ----------

# Look at subsession hours > 40
highssl_day = ms_1day_raw.where("subsession_hours > 40")
display(highssl_day)

# COMMAND ----------

# Look at summary stats for subsession hours > 40
display(highssl_day.describe())

# COMMAND ----------

display(ms_1day_raw.describe())

# COMMAND ----------


