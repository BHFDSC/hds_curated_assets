# Databricks notebook source
# MAGIC %md #Setup

# COMMAND ----------

# MAGIC %run ./project_config

# COMMAND ----------

from hds_functions import load_table, save_table

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

import databricks.koalas as ks
import pandas as pd
import numpy as np

import re
import io
import datetime

# COMMAND ----------

db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'
dsa = f'dsa_391419_j3w9t_collab'

# COMMAND ----------

# MAGIC %md #Data

# COMMAND ----------

# DBTITLE 1,KPCs
date_of_birth_individual = (
    load_table('date_of_birth_individual')
    .select('person_id', 'date_of_birth', 'date_of_birth_tie_flag')
)

sex_individual = (
    load_table('sex_individual')
    .select('person_id', 'sex_code', 'sex', 'sex_tie_flag')
)

ethnicity_individual = (
    load_table('ethnicity_individual')
    .select(
        'person_id', 'ethnicity_raw_code', 'ethnicity_raw_description',
        'ethnicity_18_code', 'ethnicity_18_group', 'ethnicity_5_group', 
        'ethnicity_18_tie_flag', 'ethnicity_5_tie_flag'
    )
)

lsoa_individual = (
    load_table('lsoa_individual')
    .select('person_id', 'lsoa', 'lsoa_tie_flag')
)

# COMMAND ----------

# DBTITLE 1,Lookups
imd_lookup = (
    spark.table(f'dsa_391419_j3w9t_collab.hds_curated_assets_lsoa_imd_lookup')
    .filter(f.col("LSOA_YEAR")==2011)
    .filter(f.col("IMD_YEAR")==2019)
    .select(f.col("LSOA").alias("lsoa"),
            f.col("DECILE").alias("imd_decile"),
            f.col("QUINTILE").alias("imd_quintile")
            )
    )

region_lookup = (
    spark.table(f'dsa_391419_j3w9t_collab.hds_curated_assets_lsoa_region_lookup')
    .select(f.col("lsoa_code").alias("lsoa"),
            f.col("region_name").alias("region")
            )
    )

# COMMAND ----------

# DBTITLE 1,Deaths
deaths = (
    load_table('deaths_single')
    .select('person_id', 'date_of_death')
    .withColumn('death_flag', f.lit(1))
)

# COMMAND ----------

# DBTITLE 1,GDPPR Flag
gdppr_summary = (
    load_table('gdppr', method = 'gdppr')
    .select('person_id', 'date')
    .filter("(person_id IS NOT NULL)")
    .groupBy('person_id')
    .agg(
        f.min('date').alias('gdppr_min_date'),
        f.lit(1).alias('in_gdppr')
    )
)

# COMMAND ----------

# MAGIC %md #Combine

# COMMAND ----------

final = (
    date_of_birth_individual
    .join(sex_individual,on="person_id",how="full")
    .join(ethnicity_individual,on="person_id",how="full")
    .join(lsoa_individual,on="person_id",how="full")

    # Nulls introduced due to LSOAs that are in lookups but not in EHRs
    .join(region_lookup,on="lsoa",how="full")
    .join(imd_lookup,on="lsoa",how="full")
    .filter(f.col("person_id").isNotNull())

    .join(deaths, on="person_id",how="full")
    .join(gdppr_summary,on="person_id",how="full")

    .select(
        "person_id",
        "date_of_birth",
        "sex_code",
        "sex",
        "ethnicity_raw_code",
        "ethnicity_raw_description",
        "ethnicity_18_code",
        "ethnicity_18_group",
        "ethnicity_5_group",
        "lsoa",
        "region",
        "imd_decile",
        "imd_quintile",
        "death_flag",
        "date_of_death",
        "in_gdppr",
        "gdppr_min_date",
        "date_of_birth_tie_flag",
        "sex_tie_flag",
        "ethnicity_18_tie_flag",
        "ethnicity_5_tie_flag",
        "lsoa_tie_flag"
        )
    
    )

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(final, "demographics")

# COMMAND ----------

# MAGIC %md #Check

# COMMAND ----------

demographics = load_table("demographics")

# COMMAND ----------

display(demographics)

# COMMAND ----------

# DBTITLE 1,Flags
display(
    demographics.groupBy("death_flag").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

display(
    demographics.groupBy("in_gdppr").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

display(
    demographics.groupBy("date_of_birth_tie_flag").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

display(
    demographics.groupBy("sex_tie_flag").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

display(
    demographics.groupBy("ethnicity_18_tie_flag").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

display(
    demographics.groupBy("ethnicity_5_tie_flag").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

display(
    demographics.groupBy("lsoa_tie_flag").count()
    .withColumn("pct", f.round((f.col("count") / (demographics.count())) * 100, 4))
)

# COMMAND ----------

# DBTITLE 1,Number of rows = Number of persons
print("Number of rows:")
no_rows = demographics.count()
display(no_rows)

print("\n\nNumber of persons:")
no_persons = demographics.select("person_id").distinct().count()
display(no_persons)

print("\n\nRows = Persons")
display(no_rows==no_persons)

print("\n\nNull Person ID rows")
display(demographics.filter(f.col("person_id").isNull()).count())