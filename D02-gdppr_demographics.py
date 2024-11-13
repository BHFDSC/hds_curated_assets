# Databricks notebook source
# MAGIC %run ./project_config

# COMMAND ----------

from hds_functions import update_gdppr_demographics
update_gdppr_demographics(update_all = False)