from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from .table_management import load_table, save_table, get_archive_versions


def update_gdppr_demographics(update_all: bool = False) -> None:
    """
    Update the GDPPR demographics table for all versions or only the new versions not already in the table.

    Parameters:
        update_all (bool): If True, update all versions. If False, update only new versions.

    Returns:
        None
    """
    try:
        # Attempt to load the 'gdppr_demographics_all_versions' table to check its existence
        load_table('gdppr_demographics_all_versions')
        gdppr_demographics_exists = True
    except AnalysisException:
        # If the table does not exist, set the flag to False
        gdppr_demographics_exists = False

    # Load the main GDPPR table containing all versions
    gdppr_all_versions = load_table('gdppr_all_versions', method='gdppr')

    if gdppr_demographics_exists:
        # Load the demographics table if it exists
        gdppr_demographics_all_versions = load_table('gdppr_demographics_all_versions')
        
        # Get a list of versions from the main GDPPR table and demographics table
        all_versions = get_archive_versions(gdppr_all_versions)
        existing_versions = get_archive_versions(gdppr_demographics_all_versions)

        # Determine which versions need to be computed (i.e., those not in the demographics table)
        versions_to_compute = [v for v in all_versions if v not in existing_versions]

        if versions_to_compute:
            # Filter the main GDPPR table to include only the versions that need to be computed
            gdppr_subset = gdppr_all_versions.filter(f.col('archived_on').isin(versions_to_compute))

            # Summarise the demographics for the filtered subset
            gdppr_demographics_subset = summarise_gdppr_demographics(gdppr_subset)
            
            # Union the new summarized demographics with the existing demographics table
            gdppr_demographics_updated = gdppr_demographics_all_versions.unionByName(gdppr_demographics_subset)

            # Save the updated demographics table
            save_table(df=gdppr_demographics_updated, table='gdppr_demographics_all_versions', partition_by = 'archived_on')

    else:
        # If the demographics table does not exist or update_all is True, summarize all versions
        gdppr_demographics_all_versions = summarise_gdppr_demographics(gdppr_all_versions)

        # Save the new demographics table
        save_table(df=gdppr_demographics_all_versions, table='gdppr_demographics_all_versions', partition_by = 'archived_on')


def summarise_gdppr_demographics(gdppr: DataFrame) -> DataFrame:
    """
    Summarise the GDPPR table to distinct demographics data.

    Parameters:
        gdppr (DataFrame): The GDPPR data.

    Returns:
        DataFrame: The summarised demographics data
    """
    return (
        gdppr
        .filter(f.col('person_id').isNotNull())  # Filter out rows where person_id is null
        .select(
            'archived_on', 'person_id', 'practice',
            'reporting_period_end_date', 'sex', 'lsoa',
            'ethnic', 'year_month_of_birth'  # Select relevant columns
        )
        .distinct()
    )


