# Databricks notebook source
# MAGIC %run ./project_config

# COMMAND ----------

import re
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from hds_functions import load_table, save_table


# COMMAND ----------

# MAGIC %md
# MAGIC # hes_apc_diagnosis

# COMMAND ----------

hes_apc = load_table('hes_apc', method = 'hes_apc')

id_cols = ['person_id', 'epikey', 'epistart', 'epiend', 'admidate', 'disdate']
diag_cols = [col for col in list(hes_apc.columns) if re.match(r'^diag_(3|4)_\d\d$', col)]

hes_apc_diagnosis = (
    hes_apc
    .select(id_cols + diag_cols)
    .unpivot(ids = id_cols, values = diag_cols, variableColumnName = 'diag_column', valueColumnName = 'code')
    .withColumn('diag_digits', f.regexp_extract('diag_column', r'diag_(\d)_(\d+)', 1).cast('int'))
    .withColumn('diag_position', f.regexp_extract('diag_column', r'diag_(\d)_(\d+)', 2).cast('int'))
    .withColumn('code', f.regexp_replace('code', r'X$', ''))
    .withColumn('code', f.regexp_replace('code', r'[.,\-\s]', ''))
    .filter(f.col('code').isNotNull() & (f.col('code') != f.lit('')))
)

save_table(df = hes_apc_diagnosis, table = 'hes_apc_diagnosis')

# COMMAND ----------

# MAGIC %md
# MAGIC # hes_apc_procedures

# COMMAND ----------


hes_apc_otr = load_table('hes_apc_otr', method = 'hes_apc')

# Select procedure code and date columns
opdate_cols = [col for col in list(hes_apc_otr.columns) if re.match(r'^opdate_\d\d$', col)]
opertn_cols = [col for col in list(hes_apc_otr.columns) if re.match(r'^opertn_\d\d$', col)]

# Determine the number of pairs
num_pairs = len(opdate_cols)
assert len(opdate_cols) == len(opertn_cols)

# Generate the stack expression dynamically
stack_expr = ', '.join([f"{date}, {code}, {i+1}" for i, (date, code) in enumerate(zip(opdate_cols, opertn_cols))])

hes_apc_otr_stacked = (
    hes_apc_otr
    .select('person_id', 'epikey', f.expr(f'stack({num_pairs}, {stack_expr}) as (procedure_date, original_code, position)'))
    .withColumn('cleaned_code', f.regexp_replace('original_code', r'[.,\-\s\t]', ''))
    .withColumn('valid_code_3', f.when(f.col('cleaned_code').rlike('^[A-Z]\\d{2}$'), f.lit(1)))
    .withColumn('valid_code_4', f.when(f.col('cleaned_code').rlike('^[A-Z]\\d{3}$'), f.lit(1)))
)

hes_apc_procedure = (
    hes_apc_otr_stacked
    .filter("(valid_code_3 = 1) OR (valid_code_4 = 1)")
    .withColumn('code_4', f.when(f.col('valid_code_4') == f.lit(1), f.col('cleaned_code')))
    .withColumn(
        'code_3',
        f.when(f.col('valid_code_3') == f.lit(1), f.col('cleaned_code'))
        .otherwise(f.substring(f.col('code_4'), 1, 3))
    )
    .select('person_id', 'epikey', 'position', 'procedure_date', 'code_3', 'code_4')
    .unpivot(
        ids = ['person_id', 'epikey', 'position', 'procedure_date'], values = ['code_3', 'code_4'],
        variableColumnName = 'code_column', valueColumnName = 'code'
    )
    .withColumn('code_digits', f.regexp_extract('code_column', r'code_(\d)', 1).cast('int'))
    .filter('code IS NOT NULL')
    .drop('code_column')
)

hes_apc_dates = (
    load_table('hes_apc', method = 'hes_apc')
    .select('epikey', 'epistart', 'epiend', 'admidate', 'disdate')
)

hes_apc_procedure = (
    hes_apc_procedure
    .join(
        hes_apc_dates,
        on = 'epikey',
        how ='left'
    )
)

save_table(df = hes_apc_procedure, table = 'hes_apc_procedure')

# COMMAND ----------

# MAGIC %md # hes_apc_cips_episodes

# COMMAND ----------

# Load HES-APC table
hes_apc = load_table('hes_apc', method = 'hes_apc')

# Select columns
hes_apc_cips_episodes = (
    hes_apc
    .select(
        'epikey', 'person_id', 'epistart', 'epiend', 'epiorder', 'epistat',
        'admidate', 'disdate', 'procode5', 'admisorc', 'admimeth', 
        'disdest', 'dismeth',
    )
)

# Define null dates
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'epistart',
        f.when(
            (f.col('epistart') != f.to_date(f.lit('1800-01-01')))
            & (f.col('epistart') != f.to_date(f.lit('1801-01-01'))),
            f.col('epistart')
        )
    )
    .withColumn(
        'epiend',
        f.when(
            (f.col('epiend') != f.to_date(f.lit('1800-01-01')))
            & (f.col('epiend') != f.to_date(f.lit('1801-01-01'))),
            f.col('epiend')
        )
    )
    .withColumn(
        'admidate',
        f.when(
            (f.col('admidate') != f.to_date(f.lit('1800-01-01')))
            & (f.col('admidate') != f.to_date(f.lit('1801-01-01'))),
            f.col('admidate')
        )
    )
    .withColumn(
        'disdate',
        f.when(
            (f.col('disdate') != f.to_date(f.lit('1800-01-01')))
            & (f.col('disdate') != f.to_date(f.lit('1801-01-01'))),
            f.col('disdate')
        )
    )
)

# Accept epistart as admidate if admidate is missing and epiorder is 1
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'admidate_filled',
        f.when(
            f.col('admidate').isNull() 
            & f.col('epistart').isNotNull() 
            & (f.col('epiorder') == f.lit(1))
            & (~f.col('admimeth').isin(['2B', '81']))
            & (~f.col('admisorc').isin(['51', '52', '53'])),
            f.col('epistart')
        )
        .otherwise(f.col('admidate'))
    )
)

# Identify rows where epistart greater than epiend
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'epistart_gt_epiend',
        f.when(
            f.col('epistart') > f.col('epiend'),
            f.lit(1)
        )
    )
)

# Define transit flag
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'transit',
        f.when(
            (~f.col('admisorc').isin(['51', '52', '53']))
            & (~f.col('admimeth').isin(['2B', '81']))
            & (f.col('disdest').isin(['51', '52', '53'])),
            f.lit(1)
        )
        .when(
            (
                (f.col('admisorc').isin(['51', '52', '53']))
                | (f.col('admimeth').isin(['2B', '81']))
            )
            & (f.col('disdest').isin(['51', '52', '53'])),
            f.lit(2)
        )
        .when(
            (
                (f.col('admisorc').isin(['51', '52', '53']))
                | (f.col('admimeth').isin(['2B', '81']))
            )
            & (~f.col('disdest').isin(['51', '52', '53'])),
            f.lit(3)
        )
        .otherwise(0)
    )
)

# Filter for qualifying episodes 
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .filter(
        f.col('person_id').isNotNull()
        & f.col('epikey').isNotNull()
        & f.col('procode5').isNotNull()
        & f.col('epistart').isNotNull()
        & f.col('epiend').isNotNull()
        & f.col('admidate_filled').isNotNull()
        & (f.col('epistat') == f.lit(3))
        & f.col('epistart_gt_epiend').isNull()
    )
)

save_table(df = hes_apc_cips_episodes, table = 'hes_apc_cips_episodes')

# COMMAND ----------

hes_apc_cips_episodes = load_table('hes_apc_cips_episodes')

# Define transit flag
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'transit',
        f.when(
            (~f.col('admisorc').isin(['51', '52', '53']))
            & (~f.col('admimeth').isin(['2B', '81']))
            & (f.col('disdest').isin(['51', '52', '53'])),
            f.lit(1)
        )
        .when(
            (
                (f.col('admisorc').isin(['51', '52', '53']))
                | (f.col('admimeth').isin(['2B', '81']))
            )
            & (f.col('disdest').isin(['51', '52', '53'])),
            f.lit(2)
        )
        .when(
            (
                (f.col('admisorc').isin(['51', '52', '53']))
                | (f.col('admimeth').isin(['2B', '81']))
            )
            & (~f.col('disdest').isin(['51', '52', '53'])),
            f.lit(3)
        )
        .otherwise(0)
    )
)

# Provider spell grouping and ordering
window_p_spell_grouping = (
    Window.partitionBy('person_id', 'procode5')
    .orderBy('epistart', 'epiend', 'epiorder', 'transit', 'epikey')
)

# Create lag columns
hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn('previous_admidate', f.lag('admidate').over(window_p_spell_grouping))
    .withColumn('previous_epistart', f.lag('epistart').over(window_p_spell_grouping))
    .withColumn('previous_dismeth', f.lag('dismeth').over(window_p_spell_grouping))
    .withColumn('previous_epiend', f.lag('dismeth').over(window_p_spell_grouping))
)

# An episode is considered to be part of the same provider spell as the previous 
# episode if one of the following is true:
# 1. `admidate` of the current episode is the same as for the previous episode
# 2. `epistart` of the current episode is the same as for the previous episode
# 3. The method of discharge of the previuos episode is an intra-provider transfer (DISMETH is 8 or 9)
#    and the episode start date (epistart) of the current episode matches the episode end date 
#    of the previous episode (epiend)

hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'new_p_spell',
        f.when(
            (f.col('admidate') == f.col('previous_admidate')),
            f.lit(0)
        )
        .when(
            (f.col('epistart') == f.col('previous_epistart')),
            f.lit(0)
        )
        .when(
            (f.col('previous_dismeth').isin(['8', '9'])) & (f.col('epistart') == f.col('previous_epiend')),
            f.lit(0)
        )
        .otherwise(1),
    )
    .withColumn('p_spell_order', f.sum(f.col('new_p_spell')).over(window_p_spell_grouping))
    .withColumn(
        'p_spell_id',
        f.concat(
            f.col('person_id'), f.lit('-'), f.col('procode5'), f.lit('-'),
            f.col('p_spell_order')
        )
    )
)

# Calculate episode order, episode count, first and last episode flags within each provider spell
window_p_spell_id_ordered = (
    Window.partitionBy('p_spell_id')
    .orderBy('epistart', 'epiend', 'epiorder', 'transit', 'epikey')
)

window_p_spell_id_grouped = (
    Window.partitionBy('p_spell_id')
)

hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .withColumn(
        'p_spell_epiorder',
        f.row_number().over(window_p_spell_id_ordered)
    )
    .withColumn(
        'p_spell_epi_count',
        f.max(f.col('p_spell_epiorder')).over(window_p_spell_id_grouped)
    )
    .withColumn(
        'p_spell_first_episode',
        f.when(
            f.col('p_spell_epiorder') == f.lit(1),
            f.lit(1)
        )
    )
    .withColumn(
        'p_spell_last_episode',
        f.when(
            f.col('p_spell_epiorder') == f.col('p_spell_epi_count'),
            f.lit(1)
        )
    )
)

save_table(df = hes_apc_cips_episodes, table = 'hes_apc_cips_episodes')

# COMMAND ----------

hes_apc_cips_episodes = load_table('hes_apc_cips_episodes')

p_spell_first_episode = (
    hes_apc_cips_episodes
    .filter(f.col('p_spell_first_episode') == f.lit(1))
    .select(
        'person_id', 'procode5', 'p_spell_id', 'p_spell_order',
        f.col('epistart').alias('p_spell_epistart'),
        f.col('admidate').alias('p_spell_admidate'),
        f.col('admisorc').alias('p_spell_admisorc'),
        f.col('admimeth').alias('p_spell_admimeth')
    )
)

p_spell_last_episode = (
    hes_apc_cips_episodes
    .filter(f.col('p_spell_last_episode') == f.lit(1))
    .select(
        'person_id', 'procode5', 'p_spell_id', 'p_spell_order',
        f.col('epiend').alias('p_spell_epiend'),
        f.col('disdate').alias('p_spell_disdate'),
        f.col('disdest').alias('p_spell_disdest'),
        f.col('dismeth').alias('p_spell_dismeth')
    )
)

hes_apc_cips_provider_spells = (
    p_spell_first_episode
    .join(
        p_spell_last_episode,
        on = ['person_id', 'procode5', 'p_spell_order', 'p_spell_id'],
        how = 'full'
    )
)

save_table(df = hes_apc_cips_provider_spells, table = 'hes_apc_cips_provider_spells')

# COMMAND ----------

hes_apc_cips_provider_spells = load_table('hes_apc_cips_provider_spells')

# Obtain previous epiend and disdest 
window_cips_ordered = (
    Window.partitionBy('person_id')
    .orderBy('p_spell_admidate', 'p_spell_disdate', 'procode5', 'p_spell_order')
)

hes_apc_cips_provider_spells = (
    hes_apc_cips_provider_spells
    .withColumn('prev_p_spell_epiend', f.lag('p_spell_epiend').over(window_cips_ordered))
    .withColumn('prev_p_spell_disdest', f.lag('p_spell_disdest').over(window_cips_ordered))
)

# A provider spell is considered to be part of the same CIPS as the previous provider spell 
# if `epistart` is not more than 3 days later than `epiend` of the previous spell and one of 
# the following is true:
# 1. The discharge destination of the previous spell is another hospital (`disdest` is 51, 52 or 53)
# 2. The source of admission of the current spell another hospital (`admisorc` is 51, 52 or 53)
# 3. The method of admission of the current spell is a transfer ('admimeth` is 2B or 81)

hes_apc_cips_provider_spells = (
    hes_apc_cips_provider_spells
    .withColumn(
        'new_cips',
        f.when(
            (f.datediff(f.col('p_spell_epistart'), f.col('prev_p_spell_epiend')) <= f.lit(3))
            & (
                f.col('prev_p_spell_disdest').isin(['51', '52', '53'])
                | f.col('p_spell_admisorc').isin(['51', '52', '53'])
                | f.col('p_spell_admimeth').isin(['2B', '81'])
            ),
            f.lit(0)
        )
        .otherwise(1),
    )
    .withColumn('cips_order', f.sum(f.col('new_cips')).over(window_cips_ordered))
    .withColumn(
        'cips_id',
        f.concat(f.col('person_id'), f.lit('-'), f.col('cips_order'))
    )
)

# Calculate spell order, spell count, first and last spell flags within each CIPS
window_cips_id_ordered = (
    Window.partitionBy('cips_id')
    .orderBy('p_spell_admidate', 'p_spell_disdate', 'procode5', 'p_spell_order')
)

window_cips_id_grouped = (
    Window.partitionBy('cips_id')
)

hes_apc_cips_provider_spells = (
    hes_apc_cips_provider_spells
    .withColumn(
        'cips_spell_order',
        f.row_number().over(window_cips_id_ordered)
    )
    .withColumn(
        'cips_spell_count',
        f.max(f.col('cips_spell_order')).over(window_cips_id_grouped)
    )
    .withColumn(
        'cips_first_spell',
        f.when(
            f.col('cips_spell_order') == f.lit(1),
            f.lit(1)
        )
    )
    .withColumn(
        'cips_last_spell',
        f.when(
            f.col('cips_spell_order') == f.col('cips_spell_count'),
            f.lit(1)
        )
    )
)

save_table(df = hes_apc_cips_provider_spells, table = 'hes_apc_cips_provider_spells')

# COMMAND ----------

hes_apc_cips_provider_spells = load_table('hes_apc_cips_provider_spells')

cips_first_spell = (
    hes_apc_cips_provider_spells
    .filter(f.col('cips_first_spell') == f.lit(1))
    .select(
        'person_id', 'cips_id', 'cips_order',
        f.col('p_spell_epistart').alias('cips_epistart'),
        f.col('p_spell_admidate').alias('cips_admidate'),
        f.col('p_spell_admisorc').alias('cips_admisorc'),
        f.col('p_spell_admimeth').alias('cips_admimeth')
    )
)

cips_last_spell = (
    hes_apc_cips_provider_spells
    .filter(f.col('cips_last_spell') == f.lit(1))
    .select(
        'person_id', 'cips_id', 'cips_order',
        f.col('p_spell_epiend').alias('cips_epiend'),
        f.col('p_spell_disdate').alias('cips_disdate'),
        f.col('p_spell_disdest').alias('cips_disdest'),
        f.col('p_spell_dismeth').alias('cips_dismeth')
    )
)

hes_apc_cips_cips = (
    cips_first_spell
    .join(
        cips_last_spell,
        on = ['person_id', 'cips_order', 'cips_id'],
        how = 'full'
    )
)

save_table(df = hes_apc_cips_cips, table = 'hes_apc_cips_cips')

# COMMAND ----------

hes_apc_cips_provider_spells = load_table('hes_apc_cips_provider_spells')
hes_apc_cips_cips = load_table('hes_apc_cips_cips')

hes_apc_cips_provider_spells = (
    hes_apc_cips_provider_spells
    .join(
        hes_apc_cips_cips,
        on = ['person_id', 'cips_order', 'cips_id'],
        how = 'left'
    )
)

save_table(df = hes_apc_cips_provider_spells, table = 'hes_apc_cips_provider_spells')

# COMMAND ----------

hes_apc_cips_episodes = load_table('hes_apc_cips_episodes')
hes_apc_cips_provider_spells = load_table('hes_apc_cips_provider_spells')

hes_apc_cips_episodes = (
    hes_apc_cips_episodes
    .join(
        hes_apc_cips_provider_spells
        .drop(*['procode5', 'p_spell_order']),
        on = ['person_id', 'p_spell_id'],
        how = 'left'
    )
)

save_table(df = hes_apc_cips_episodes, table = 'hes_apc_cips_episodes')