# Databricks notebook source
import pyspark.sql.functions as F
import pyspark.sql.types as st
from sklearn import preprocessing
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data

# COMMAND ----------

#Global variables
sample_id = 42
week_1_start = '20180923'
week_1_end = '20180929'
week_4_end = '20181020'
week_12_end = '20181215'
week_16_end = '20190112'
day_1_date = '20180925'

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
        sample_id,
        document_id
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
        sample_id,
        document_id
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
        sample_id,
        document_id
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

ms_12week_raw = spark.sql("""
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
        reason
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
    """.format(week_1_start, week_12_end, sample_id)
    )


# COMMAND ----------

# Look at the summary stats for raw ping values over a day
display(ms_1day_raw['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'submission_date_s3'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC From search_clients_daily, group by day then add sap and in_content_organic for total daily search counts.  
# MAGIC Then join with the table of other count values summed over 1 day.

# COMMAND ----------

# Replace the null values with 0s
search_day = search_day.na.fill(0)
# Sum over the day
search_day_sum = search_day.groupBy("client_id", "submission_date_s3") \
                        .agg(F.sum(F.col("sap")+F.col("in_content_organic")))  \
                        .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts')
# Join the daily search counts with the other daily counts 
ms_1day = ms_1day_sum.join(search_day_sum, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)
display(ms_1day)

# COMMAND ----------

# Sanity check
display(search_day_sum)

# COMMAND ----------

# Look at the raw ping values over a week
display(ms_1week_raw['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'submission_date_s3'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC From search_clients_daily, group by date then add sap and in_content_organic for total daily search counts.  
# MAGIC Then join with the table of other count values summed over 1 day.

# COMMAND ----------

# Replace the null values with 0s
search_wk = search_wk.na.fill(0)
# Sum over the day
search_wk_sum = search_wk.groupBy("client_id", "submission_date_s3") \
                        .agg(F.sum(F.col("sap")+F.col("in_content_organic"))) \
                        .withColumnRenamed('sum((sap + in_content_organic))','search_counts')
search_wk_sum.describe().show()
# Note - this isn't joined with week values because we don't have week values summed by day

# COMMAND ----------

# Sanity check
display(search_wk_sum)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Start looking at some specific cases to determine if they are invalid or show us something interesting about usage.

# COMMAND ----------

# How many 1 day ping records with uri_count of zero
ms_1day_0uri = ms_1day_raw.filter(F.col("uri_count") == 0)
ms_1day_0uri.count()

# COMMAND ----------

# How many 1 day ping records with active ticks of zero
ms_1day_0at = ms_1day_raw.filter(F.col("active_ticks") == 0)
ms_1day_0at.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Lots of pings with 0 uri and 0 active ticks.

# COMMAND ----------

# How many 1 day ping records with both uri and active tick of 0
ms_1day_p_0uriat = ms_1day_raw.filter((F.col("uri_count") == 0) & (F.col("active_ticks") == 0))
ms_1day_p_0uriat.count()

# COMMAND ----------

# Look at pings with 0 uri and 0 active ticks
display(ms_1day_p_0uriat)

# COMMAND ----------

# Number of distinct client ids with both 0 uri and 0 active_ticks
ms_1day_p_0uriat.select("client_id").distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC How many fewer clients with both zero counts would there be if the pings were consolidated into days?  
# MAGIC Still a lot

# COMMAND ----------

# Number of client recods in a day with both uri and active ticks of 0
ms_1day_s_0uriat = ms_1day.filter((F.col("td_uri") == 0) & (F.col("td_active_ticks") == 0))
ms_1day_s_0uriat.count()

# COMMAND ----------

# Look at the specific records
display(ms_1day_s_0uriat)

# COMMAND ----------

# MAGIC %md
# MAGIC How many clients with total daily uri >= 5 with 0 active_ticks?  
# MAGIC Greater than 72,000 - Not sure how it is possible to get this usage pattern.

# COMMAND ----------

# Number of client recods in a day with uri >= 5 and active ticks of 0
ms_1day_s_uri_0at = ms_1day_sum.where("td_uri >= 5 and td_active_ticks = 0")
ms_1day_s_uri_0at.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Double check 0 uri and active ticks for client 5f4aaf58-8b0c-44a5-8208-6e64dd36e9ec with 32 search count  
# MAGIC Lots of short sessions in 1 day, profile_subsession_counter in order, but no activity

# COMMAND ----------

# Look at 1 day's pings for client e9ec
ms_1day_test = ms_1day_raw.filter("client_id = '5f4aaf58-8b0c-44a5-8208-6e64dd36e9ec'")
display(ms_1day_test)

# COMMAND ----------

# Check search counts for the same client as above, e9ec
search_day_test = search_day.filter("client_id = '5f4aaf58-8b0c-44a5-8208-6e64dd36e9ec'")
display(search_day_test)

# COMMAND ----------

# MAGIC %md
# MAGIC A client found in the table above with 0 uri and active ticks but 24 subsession hours for 1 day.   
# MAGIC How do they look over a week and for search?  Same pattern for 7 days and no search counts.

# COMMAND ----------

# One day's pings for client 68a8
ms_1day_test = ms_1day_raw.filter("client_id = '014b77b1-b74f-4fe4-ade9-f492119968a8'")
display(ms_1day_test)

# COMMAND ----------

# One week's pings for client 68a8
ms_1week_test = ms_1week_raw.filter("client_id = '014b77b1-b74f-4fe4-ade9-f492119968a8'")
display(ms_1week_test)

# COMMAND ----------

# Search counts for the week
search_week_test = search_wk.filter("client_id = '014b77b1-b74f-4fe4-ade9-f492119968a8'")
display(search_week_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Subsession Length is 0

# COMMAND ----------

ms_1day_ssl0 = ms_1day_raw.filter((F.col("subsession_length") == 0))
ms_1day_ssl0.count()
# Small number of pings

# COMMAND ----------

# Look at the details of these pings
display(ms_1day_ssl0)

# COMMAND ----------

# Summary stats of pings for 1 day
display(ms_1day_ssl0['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'session_length', 'submission_date_s3'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Not sure why the uri_count and active_ticks are occasionally > 0 for subsession length of 0

# COMMAND ----------

# Look at ping with subsession length 0 and active ticks > 20
ssl0 = ms_1day_ssl0.where("active_ticks > 20")
display(ssl0)

# COMMAND ----------

# Check if this ssl0 client has any search counts
search_day_test = search_day.filter("client_id = 'becb9c30-48a8-4055-bc60-2e1b9e07d808'")
display(search_day_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Check what this ssl0 client looks like for the rest of the week  
# MAGIC The other pings look normal.  
# MAGIC profile_subsession_counter 908 is a shutdown ping with the same session_length as 907.  Perhaps these uri and active tick counts are really from the previous subsession?

# COMMAND ----------

# Check what this ssl0 client looks like for the rest of the week
ms_1week_test = ms_1week_raw.filter("client_id = 'becb9c30-48a8-4055-bc60-2e1b9e07d808'")
display(ms_1week_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### High Active Ticks

# COMMAND ----------

# Summary stats for pings with active ticks greater than 1440
ms_1day_high_at = ms_1day_raw.filter(F.col("active_ticks") > 1440)
ms_1day_high_at['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours'].describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at one client's use over a week for the highest active_ticks value.  
# MAGIC Is this a bot?  0 uri, active 24 hours 7 days a week, in Moscow.

# COMMAND ----------

# Look at client bcc2 over 1 week of pings
highat_c = ms_1week_raw.where("client_id ='ec05d93d-c4a8-4b07-8930-f1407e1fbcc2'").sort("submission_date_s3")
display(highat_c)

# COMMAND ----------

# MAGIC %md
# MAGIC The usage is 24 hours every day.  These numbers look a bit different than that because there is 1 day with 2 subsessions.

# COMMAND ----------

# Summary stats for client bcc2 over a week
highat_c['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours'].describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Does this pattern hold up over more than 1 week?  
# MAGIC Yes over 4 weeks

# COMMAND ----------

# Look at client bcc2 over 4 weeks of pings
highat_c4w = ms_4week_raw.where("client_id ='ec05d93d-c4a8-4b07-8930-f1407e1fbcc2'").sort("profile_subsession_counter", "submission_date_s3")
display(highat_c4w)

# COMMAND ----------

# Summary stats for 4 weeks for client bcc2
highat_c4w['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours'].describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC  0 uri, active 24 hours not every day for 12 weeks but the majority of them

# COMMAND ----------

# Look at client bcc2 over 12 weeks of pings
highat_c12w = ms_12week_raw.where("client_id ='ec05d93d-c4a8-4b07-8930-f1407e1fbcc2'").sort("profile_subsession_counter", "submission_date_s3")
display(highat_c12w)

# COMMAND ----------

# Look at pings over 12 weeks where uri count is not 0
display(highat_c12w.where("uri_count > 0"))

# COMMAND ----------

# Summary stats for 12 weeks for 1 client id
highat_c12w['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours'].describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #####How many other clients have high active ticks and 0 uri in 1 week?

# COMMAND ----------

# Look at all clients over a week of pings where uri count is 0 and active ticks hours > 23
highat_w = ms_1week_raw.filter((F.col("uri_count") == 0) & (F.col("active_tick_hrs") > 23))
display(highat_w)

# COMMAND ----------

# MAGIC %md
# MAGIC 18 distinct clients

# COMMAND ----------

# Count distinct clients over a week where uri count is 0 and active ticks hours > 23
highat_clients = highat_w.select("client_id").distinct()
highat_clients.count()

# COMMAND ----------

# Look at client ids where uri count is 0 and active ticks hours > 23
display(highat_clients)

# COMMAND ----------

# MAGIC %md
# MAGIC Look at some of the additional clients

# COMMAND ----------

# MAGIC %md
# MAGIC This client looks like "normal"/mixed usage.  
# MAGIC Only one day of 0 uri and 24 hrs active, profile_subsession_counter in order, some short subsessions, some high active ticks with non-zero uris.

# COMMAND ----------

# Look at 1 week of pings for client 6a8e
highat_c1 = ms_1week_raw.where("client_id ='6693ef5a-7ae3-424e-b59c-2995a7d66a8e'").sort("submission_date_s3")
display(highat_c1)

# COMMAND ----------

# MAGIC %md
# MAGIC Low search counts, but not 0

# COMMAND ----------

# Look at search counts for client 6a8e
search_wk_test = search_wk.filter("client_id = '6693ef5a-7ae3-424e-b59c-2995a7d66a8e'")
display(search_wk_test)

# COMMAND ----------

# MAGIC %md
# MAGIC This second client also has "normal" usage - 7 daily pings, profile_subsession_counter in order, short active tick hours and only 1 odd record with 0 uri and 24 active tick hours

# COMMAND ----------

# Look at 1 week of pings for client 395e
highat_c2 = ms_1week_raw.where("client_id ='5320032f-bf72-4e4e-a8c3-fdf98a07395e'").sort("submission_date_s3")
display(highat_c2)

# COMMAND ----------

# Look at search counts for client 395e
search_wk_test = search_wk.filter("client_id = '5320032f-bf72-4e4e-a8c3-fdf98a07395e'")
display(search_wk_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Bot?  
# MAGIC Most of the records for the week are 0 uri 24 active hours.  profile_subsession_counter 159 is missing which could account for the short time on 9/23

# COMMAND ----------

# Look at 1 week of pings for client 0f8f
highat_c3 = ms_1week_raw.where("client_id ='4c522c98-9aff-4ba0-9040-536100560f8f'").sort("submission_date_s3")
display(highat_c3)

# COMMAND ----------

# Summary stats for 1 week of client 0f8f
display(highat_c3['uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'session_length', 'submission_date_s3', 'country', 'city'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Bot?  
# MAGIC 7 very similar days

# COMMAND ----------

 # Look at 1 week of pings for client cdec
highat_c4 = ms_1week_raw.where("client_id ='61d864f6-423d-4a97-8f10-c86d7f20cdec'").sort("submission_date_s3")
display(highat_c4)

# COMMAND ----------

# MAGIC %md
# MAGIC Bot?  
# MAGIC 7 very similar days

# COMMAND ----------

# 1 week of pings for client 715e
highat_c5 = ms_1week_raw.where("client_id ='e2e76a8a-5c3c-4978-bd73-94681e62715e'").sort("submission_date_s3")
display(highat_c5)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Client ID used on multiple machines?

# COMMAND ----------

# check for duplicates of client id, submission date and profile_subsession_counter in 1 day of pings
ms_1day_dup = ms_1day_raw.groupBy('client_id', 'submission_date_s3', 'profile_subsession_counter')\
    .count()\
    .where(F.col('count') > 1).sort('client_id', 'submission_date_s3', 'profile_subsession_counter')
display(ms_1day_dup)

# COMMAND ----------

# Summary stats for duplicate client_id and profile_subsession_counter in raw pings in 1 day
ms_1day_dup.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC There are still a lot of these duplicate client_id and profile_subsession_counter in records with at least 1 uri.

# COMMAND ----------

# Get 1day raw where uri > 1
ms_1day_1uri = ms_1day_raw.where("uri_count >= 1")
# check for duplicates of client id, submission date and profile_subsession_counter in 1 day of pings
ms_1day_1uri_dup = ms_1day_1uri.groupBy('client_id', 'submission_date_s3', 'profile_subsession_counter')\
    .count()\
    .where(F.col('count') > 1).sort('client_id', 'submission_date_s3', 'profile_subsession_counter')
# Summary stats for duplicate client_id and profile_subsession_counter in 1 day
ms_1day_1uri_dup.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Look at client with the highest number of duplicate profile_subsession_counters in 1 day

# COMMAND ----------

# Look at client with highest number of duplicates
dup_high = ms_1day_dup.where("count = 1157")
display(dup_high)

# COMMAND ----------

# MAGIC %md
# MAGIC The following client has a lot of records in one day, some Windows some Linux.  Many of them (like profile_subsession_counter = 15) look like duplicate entries but they have different client_submission_dates.  Would private browsing cause all 0s and nulls?

# COMMAND ----------

# Show all the pings in a day for client 038e
dup_c = ms_1day_raw.where("client_id ='d03252e7-4d4c-45fd-8772-223eb500038e'").sort("submission_date_s3", "profile_subsession_counter")
display(dup_c)

# COMMAND ----------

# Show summary stats for client 038e with duplicate profile_subsession_counters in a day
display(dup_c.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Lots of different users using this client_id  
# MAGIC profile_subsession_counter reused many times on one day, different operating systems, in many different locations

# COMMAND ----------

# Show duplicate counts by profile_subsession_counter for client 038e over the day
dup_c_count = dup_c.groupBy("submission_date_s3", "profile_subsession_counter") \
  .agg(F.count(F.lit(1)).alias("num_records"), F.min("os"))
display(dup_c_count)

# COMMAND ----------

# Show countries for client 038e
dup_c.select("country",  "city").distinct().show()

# COMMAND ----------

# Look at profile_subsession_counter 15 for client 038e
dup_c15 = dup_c.where("profile_subsession_counter = 15")
display(dup_c15.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Client 7728 has lots of identical looking rows for profile_subsession_counter = 35 and 36 all on the same day. Different client_submission_dates, but same session_id and subsession_id.  ??

# COMMAND ----------

# profile_subsession_counter 35 for client 7728
dup_c35 = ms_1day_raw.where("client_id ='006c8891-60c4-42ff-8f32-4cb365627728' and profile_subsession_counter = 35").sort("submission_date_s3", "profile_subsession_counter")
display(dup_c35)

# COMMAND ----------

# profile_subsession_counter 35 for client 7728
display(dup_c35.describe())

# COMMAND ----------

# profile_subsession_counter 36 for client 7728
dup_c36 = ms_1day_raw.where("client_id ='006c8891-60c4-42ff-8f32-4cb365627728' and profile_subsession_counter = 36").sort("submission_date_s3", "profile_subsession_counter")
display(dup_c36)

# COMMAND ----------

# profile_subsession_counter 36 for client 7728
display(dup_c36.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC What's going on with client 7728 over a week?  
# MAGIC Lots of duplicate profile_subsession_counters over different days.  Records looks like copies except for client_submission_date

# COMMAND ----------

# Look at a week of pings for client 7728
dup_c1 = ms_1week_raw.where("client_id ='006c8891-60c4-42ff-8f32-4cb365627728'").sort("submission_date_s3", "profile_subsession_counter")
display(dup_c1)

# COMMAND ----------

# summary stats for client 7728 for week
display(dup_c1['client_id', 'uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'profile_subsession_counter', 'submission_date_s3', 'country'].describe())

# COMMAND ----------

# Show duplicate counts by profile_subsession_counter for client 7728 over the week
dup_c1_count = dup_c1.groupBy("submission_date_s3", "profile_subsession_counter") \
  .agg(F.count(F.lit(1)).alias("num_records"), F.min("os")) \
  . sort("submission_date_s3", "profile_subsession_counter")
display(dup_c1_count)

# COMMAND ----------

#Look at profile counter 35 for client 7728 for 9/24
dup_c1_3524 = dup_c1.where("profile_subsession_counter = 35 and submission_date_s3 = '20180924'").sort("submission_date_s3", "profile_subsession_counter")
display(dup_c1_3524['client_id', 'uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'profile_subsession_counter', 'submission_date_s3', 'client_submission_date', 'country'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC There are lots of records in one week for client c62d.  Any patterns?  
# MAGIC Mostly 0 uri and active ticks.  Tab count and client submission date are often null.

# COMMAND ----------

# One week of pings for client c62d
diff_psc_c1 = ms_1week_raw.where("client_id ='9875b3d4-50cd-46aa-93c1-f2a01cedc62d'").sort("submission_date_s3", "profile_subsession_counter")
display(diff_psc_c1)

# COMMAND ----------

# Summary stats for 1 week of pings for client c62d
display(diff_psc_c1['client_id', 'uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'profile_subsession_counter', 'submission_date_s3', 'tab_count', 'client_submission_date', 'country'].describe())

# COMMAND ----------

# All the countries for client c62d
diff_psc_c1.select("country").distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC One profile subsession counter for client c62d

# COMMAND ----------

# Look at one profile_subsession_counter
diff_psc_c12 = diff_psc_c1.where("profile_subsession_counter = 12").sort("submission_date_s3", "profile_subsession_counter")
display(diff_psc_c12)

# COMMAND ----------

# Summary stats for profile counter 12 for client c62d
display(diff_psc_c12['client_id', 'uri_count', 'active_ticks', 'active_tick_hrs', 'subsession_length', 'subsession_hours', 'profile_subsession_counter', 'submission_date_s3', 'country'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Look at pings for client c62d where the active ticks > 10

# COMMAND ----------

# Active ticks > 10
diff_psc_high = diff_psc_c1.where("active_ticks > 10").sort("submission_date_s3", "profile_subsession_counter")
display(diff_psc_high)

# COMMAND ----------

# Summary stats for client c62d with active ticks > 10
display(diff_psc_high.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Clearly different users with same client_id  
# MAGIC Profile subsession counters don't increment

# COMMAND ----------

# week of pings for client e43c
diff_psc_c2 = ms_1week_raw.where("client_id ='001dcdc4-fb41-4c82-be5d-5d25a3e4e43c'").sort("submission_date_s3", "profile_subsession_counter")
display(diff_psc_c2)

# COMMAND ----------

# Summary stats for 1 week of client e43c
display(diff_psc_c2.describe())

# COMMAND ----------

# All the cities in Germany for this client e43c
display(diff_psc_c2.select("country", "city").distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ####High URI count, 0 active ticks

# COMMAND ----------

# Look at client with high uri count and 0 active ticks
ms_1week_check = ms_1week_raw.where("client_id = 'd5b70fa4-31df-48a6-863f-c0636c4d3afd'")
display(ms_1week_check)
# This pattern holds for multiple days, reasonable subsession_length, reasonable domains_count

# COMMAND ----------

# Look at client ^ for 4 weeks
ms_4week_check = ms_4week_raw.where("client_id = 'd5b70fa4-31df-48a6-863f-c0636c4d3afd'")
display(ms_4week_check)
# Same pattern over 4 weeks, profile_subsession_counter in order

# COMMAND ----------

# How many clients in 1 week have this pattern?
ms_1week_highuri_0at = ms_1week_raw.where("uri_count > 1000 and active_ticks = 0").sort("client_id")
display(ms_1week_highuri_0at)

# COMMAND ----------

# Lots of clients in 1 week with high uri and 0 active ticks
ms_1week_check.select("client_id").distinct().count()

# COMMAND ----------

# Look at how many pings each client has in a week with high uri and 0 active ticks
ms_1week_highuri_0at_count = ms_1week_highuri_0at.groupBy('client_id').count()
display(ms_1week_highuri_0at_count)

# COMMAND ----------


