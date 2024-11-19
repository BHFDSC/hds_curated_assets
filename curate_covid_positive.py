"""
Module name: curate_covid_positive.py

Description:
    This module provides functions to curate and consolidate COVID-19 positive case records 
    from multiple data sources. It extracts relevant data from various sources and saves a 
    consolidated table. The supported data sources include SGSS, Pillar 2 antigen testing,
    GDPPR, HES-APC (diagnosis), and CHESS.

"""

from pyspark.sql import functions as f, DataFrame
from functools import reduce
from typing import List
from .environment_utils import get_spark_session
from .table_management import load_table, save_table
from .csv_utils import read_csv_file


def create_covid_positive_table(table_key: str = 'covid_positive', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing positive COVID-19 records from multiple sources and save it to a table.

    Args:
        table_key (str, optional): The table key of the table to save the consolidated data.
            Defaults to 'covid_positive'.
        extraction_methods (List[str], optional): List of methods for extracting positive COVID-19 records.
            Defaults to None.
            Allowed values are: 'sgss', 'pillar_2', 'gdppr', 'hes_apc_any_diagnosis', 'hes_apc_primary_diagnosis', 'chess'.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['sgss', 'pillar_2', 'gdppr', 'hes_apc_any_diagnosis', 'hes_apc_primary_diagnosis', 'chess']

    # Get spark session
    spark = get_spark_session()

    # Extract COVID-19 records from multiple sources
    covid_positive_from_sources = [extract_covid_positive_records(method) for method in extraction_methods]
    covid_positive = reduce(DataFrame.unionByName, covid_positive_from_sources)

    # Save the consolidated data to a table
    save_table(covid_positive, table_key)

    # Clear cache
    spark.catalog.clearCache()


def extract_covid_positive_records(extract_method: str) -> DataFrame:
    """
    Extract covid positive cases based on the specified data source.

    Args:
        extract_method (str): The data source to extract sex data from.
            Allowed values are: 'sgss', 'pillar_2', 'gdppr', 'hes_apc_any_diagnosis', 'hes_apc_primary_diagnosis', 'chess'.

    Returns:
        DataFrame: DataFrame containing the extracted COVID-19 positive records.
            The DataFrame includes columns for person ID, record date, code, covid status and data source.

    """

    extraction_methods = {
        'sgss': {
            'extraction_function': covid_positive_from_sgss,
            'data_source': 'sgss',
            'load_method': 'sgss'
        },
        'pillar_2': {
            'extraction_function': covid_positive_from_pillar_2,
            'data_source': 'pillar_2',
            'load_method': 'pillar_2'
        },
        'gdppr': {
            'extraction_function': covid_positive_from_gdppr,
            'data_source': 'gdppr',
            'load_method': 'gdppr'
        },
        'hes_apc_any_diagnosis': {
            'extraction_function': covid_positive_from_hes_apc_any_diagnosis,
            'data_source': 'hes_apc_diagnosis',
            'load_method': None
        },
        'hes_apc_primary_diagnosis': {
            'extraction_function': covid_positive_from_hes_apc_primary_diagnosis,
            'data_source': 'hes_apc_diagnosis',
            'load_method': None
        },
        'chess': {
            'extraction_function': covid_positive_from_chess,
            'data_source': 'chess',
            'load_method': 'chess'
        }
    }

    if extract_method not in extraction_methods:
        raise ValueError(
            f"Invalid extract_method: {extract_method}. Allowed values are: 'sgss', 'pillar_2', 'gdppr', "
            "'hes_apc_any_diagnosis', 'hes_apc_primary_diagnosis', 'chess'."
        )

    return extraction_methods[extract_method]['extraction_function'](
        load_table(
            table=extraction_methods[extract_method]['data_source'],
            method=extraction_methods[extract_method]['load_method']
        )
    )


def covid_positive_from_sgss(sgss: DataFrame) -> DataFrame:
    """
    Extract COVID-19 positive records from the SGSS table, ensuring distinct records.

    Args:
        sgss (DataFrame): DataFrame containing SGSS data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    covid_positive_sgss = (
        sgss
        .select('person_id', 'specimen_date', 'lab_report_date')
        .withColumn(
            'date',
            f.when(
                f.col('specimen_date').isNotNull(), f.col('specimen_date')
            )
            .otherwise(f.col('lab_report_date'))
        )
        .filter("(person_id IS NOT NULL) AND (date IS NOT NULL)")
        .select('person_id', 'date')
        .distinct()
        .withColumn('code', f.lit(None))
        .withColumn('description', f.lit(None))
        .withColumn('covid_status', f.lit('confirmed'))
        .withColumn('data_source', f.lit('sgss'))
    )

    return covid_positive_sgss


def covid_positive_from_pillar_2(pillar_2: DataFrame) -> DataFrame:
    """
    Extract COVID-19 positive records from the COVID-19 antigen testing (Pillar 2) table, ensuring distinct records.

    Args:
        pillar_2 (DataFrame): DataFrame containing COVID-19 antigen testing (Pillar 2) data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    covid_19_antigen_testing_snomed = (
        read_csv_file('./codelists/covid_19_antigen_testing_snomed.csv')
        .select('code', 'description', 'test_result')
    )

    covid_positive_pillar_2 = (
        pillar_2
        .select('person_id', f.to_date('teststartdate').alias('date'), f.col('testresult').alias('test_result_original'))
        .withColumn(
            'code',
            f.when(
                f.col('test_result_original').rlike("SCT:\\d+"),
                f.regexp_extract('test_result_original', "SCT:(\\d+)", 1)
            )
            .otherwise(None)
        )
        .join(
            f.broadcast(covid_19_antigen_testing_snomed),
            on = 'code', how = 'left'
        )
    )

    covid_positive_pillar_2.cache()

    covid_positive_pillar_2 = (
        covid_positive_pillar_2
        .withColumn(
            'test_result',
            f.when(f.col('test_result').isNull(), f.col('test_result_original'))
            .otherwise(f.col('test_result'))
        )
        .filter("(person_id IS NOT NULL) AND (date IS NOT NULL) AND (test_result = 'Positive')")
        .select('person_id', 'date', 'code', 'description')
        .distinct()
        .withColumn('covid_status', f.lit('confirmed'))
        .withColumn('data_source', f.lit('pillar_2'))
    )

    return covid_positive_pillar_2


def covid_positive_from_gdppr(gdppr: DataFrame) -> DataFrame:
    """
    Extract COVID-19 positive records from the GDPPR table, ensuring distinct records.

    Args:
        gdppr (DataFrame): DataFrame containing GDPPR demographics data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    covid_19_infection_snomed_codelist = (
        read_csv_file("./codelists/covid_19_infection_snomed.csv")
        .select('code', 'description', 'covid_status')
    )

    covid_positive_gdppr = (
        gdppr
        .select('person_id', 'date', 'record_date', 'code')
        .withColumn('date', f.when(f.col('date').isNull(), f.col('record_date')).otherwise(f.col('date')))
        .join(
            f.broadcast(covid_19_infection_snomed_codelist),
            on = 'code', how = 'inner'
        )
    )

    covid_positive_gdppr.cache()

    covid_positive_gdppr = (
        covid_positive_gdppr
        .filter("(person_id IS NOT NULL) AND (date IS NOT NULL)")
        .select('person_id', 'date', 'code', 'description', 'covid_status')
        .distinct()
        .withColumn('data_source', f.lit('gdppr'))
    )

    return covid_positive_gdppr


def covid_positive_from_hes_apc_any_diagnosis(hes_apc_diagnosis: DataFrame) -> DataFrame:
    """
    Extract COVID-19 positive records from the HES-APC diagnosis table, accepting any diagnosis position,
    and ensuring distinct records.

    Args:
        hes_apc_diagnosis (DataFrame): DataFrame containing HES-APC diagnosis table.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    covid_19_infection_icd10_codelist = (
        read_csv_file("./codelists/covid_19_infection_icd10.csv")
        .select('code', 'description', 'covid_status')
    )

    covid_positive_hes_apc_matched = (
        hes_apc_diagnosis
        .select('person_id', f.col('epistart').alias('date'), 'code')
        .join(
            f.broadcast(covid_19_infection_icd10_codelist),
            on = 'code', how = 'inner'
        )
    )

    covid_positive_hes_apc_matched.cache()

    covid_positive_hes_apc_any_diagnosis = (
        covid_positive_hes_apc_matched
        .filter("(person_id IS NOT NULL) AND (date IS NOT NULL)")
        .select('person_id', 'date', 'code', 'description', 'covid_status')
        .distinct()
        .withColumn('data_source', f.lit('hes_apc_any_diagnosis'))
    )

    return covid_positive_hes_apc_any_diagnosis


def covid_positive_from_hes_apc_primary_diagnosis(hes_apc_diagnosis: DataFrame) -> DataFrame:
    """
    Extract COVID-19 positive records from the HES-APC diagnosis table, restricting to primary diagnosis,
    and ensuring distinct records.

    Args:
        hes_apc_diagnosis (DataFrame): DataFrame containing HES-APC diagnosis table.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    covid_19_infection_icd10_codelist = (
        read_csv_file("./codelists/covid_19_infection_icd10.csv")
        .select('code', 'description', 'covid_status')
    )

    covid_positive_hes_apc_matched = (
        hes_apc_diagnosis
        .select('person_id', f.col('epistart').alias('date'), 'code', 'diag_position')
        .filter("diag_position = 1")
        .join(
            f.broadcast(covid_19_infection_icd10_codelist),
            on = 'code', how = 'inner'
        )
    )

    covid_positive_hes_apc_matched.cache()

    covid_positive_hes_apc_primary_diagnosis = (
        covid_positive_hes_apc_matched
        .filter("(person_id IS NOT NULL) AND (date IS NOT NULL)")
        .select('person_id', 'date', 'code', 'description', 'covid_status')
        .distinct()
        .withColumn('data_source', f.lit('hes_apc_primary_diagnosis'))
    )

    return covid_positive_hes_apc_primary_diagnosis


def covid_positive_from_chess(chess: DataFrame) -> DataFrame:
    """
    Extract COVID-19 positive records from the CHESS table, ensuring distinct records.

    Args:
        chess (DataFrame): DataFrame containing HES-APC diagnosis data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    covid_positive_chess = (
        chess
        .select('person_id', f.col('hospitaladmissiondate').alias('date'), 'covid19')
        .filter("(person_id IS NOT NULL) AND (date IS NOT NULL) AND (covid19 = 'Yes')")
        .select('person_id', 'date')
        .distinct()
        .withColumn('code', f.lit(None))
        .withColumn('description', f.lit(None))
        .withColumn('covid_status', f.lit('confirmed'))
        .withColumn('data_source', f.lit('chess'))
    )

    return covid_positive_chess