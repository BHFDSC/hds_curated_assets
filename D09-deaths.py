# Databricks notebook source
# MAGIC %run ./project_config

# COMMAND ----------

import re
from pyspark.sql import functions as f
from hds_functions import load_table, save_table, first_row

deaths = load_table('deaths', method = 'deaths')

# COMMAND ----------

deaths_single = (
    deaths
    .filter(f.col('person_id').isNotNull())
    .transform(
        first_row,
        partition_by = ['person_id'],
        order_by = [
            f.col('reg_date').asc_nulls_last(),
            f.col('date_of_death').asc_nulls_last(),
            f.col('s_underlying_cod_icd10').asc_nulls_last()
        ]
    )
)

save_table(df = deaths_single, table = 'deaths_single')

# COMMAND ----------


deaths_single = load_table('deaths_single')
s_cod_cols =  [col for col in list(deaths_single.columns) if re.match(r'^s_cod_code_\d(\d)*$', col)]

deaths_cause_of_death = (
    deaths_single
    .select(['person_id', 'date_of_death', 's_underlying_cod_icd10'] + s_cod_cols)
    .unpivot(
        ids = ['person_id', 'date_of_death'],
        values = ['s_underlying_cod_icd10'] + s_cod_cols,
        variableColumnName = 'cod_position',
        valueColumnName = 'code_4'
    )
    .withColumn(
        'cod_position',
        f.when(
            f.col('cod_position') == f.lit('s_underlying_cod_icd10'),
            f.lit('underlying')
        )
        .otherwise(f.regexp_replace('cod_position', 's_cod_code_', 'contributory_'))
    )
    .withColumn('code_3', f.substring(f.col('code_4'), 1, 3))
    .unpivot(
        ids = ['person_id', 'date_of_death', 'cod_position'],
        values = ['code_3', 'code_4'],
        variableColumnName = 'cod_digits',
        valueColumnName = 'code'
    )
    .withColumn('cod_digits', f.regexp_extract('cod_digits', r'code_(\d)', 1).cast('int'))
    .withColumn('code', f.regexp_replace('code', r'[.,\-\s]', ''))
    .filter(f.col('code').isNotNull() & (f.col('code') != f.lit('')))
)

save_table(df = deaths_cause_of_death, table = 'deaths_cause_of_death')