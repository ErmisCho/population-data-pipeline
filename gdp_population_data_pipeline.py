import requests
import os
import logging
import sys
import json
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import configparser


def load_config(config_file="config.ini"):
    """
    Loads configuration settings from an INI file using ConfigParser.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        ConfigParser object: Parsed configuration settings.
    """
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        return config
    except Exception as e:
        logging.error(f"Failed to load config file {config_file}: {e}")
        sys.exit(1)


def download_url(url, save_path, chunk_size=128):
    """
    Downloads a file from a given URL and saves it to a specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The file path where the downloaded file will be saved.
        chunk_size (int, optional): The chunk size to use while downloading. Defaults to 128.
    """
    try:
        r = requests.get(url, stream=True)
        # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        r.raise_for_status()
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return False


def read_data(output_path):
    """
    Reads CSV files from ZIP archives located in the specified output directory.

    Args:
        output_path (str): The directory path where the ZIP files are located.

    Returns:
        list: A list of pandas DataFrames, one for each dataset (in the order of GDP and Population).

    Raises:
        SystemExit: If the required datasets are not found.
    """
    logging.info("Reading data ...")
    files = ['gdp.zip', 'population.zip']
    dfs = []

    for file in files:
        zip_file = ZipFile(os.path.join(output_path, file))
        metadata_keyword = "Metadata"
        try:
            for file in zip_file.infolist():
                if not file.filename.startswith(metadata_keyword):
                    dfs.append(pd.read_csv(zip_file.open(
                        file.filename), skiprows=3))
                    break
        except Exception as e:
            logging.error(f"Failed to read file: {e}")
            sys.exit(1)

    if len(dfs) != 2:
        logging.error(
            "Could not collect both excel files related to GDP and Population")
        sys.exit(1)
    logging.info("Read data successfully!")
    return dfs


def process_data(dfs):
    """
    Processes GDP and Population data to calculate GDP per capita and its yearly percentage change.
    The data is then reshaped into a long format.

    Args:
        dfs (list): A list of pandas DataFrames containing GDP and Population data.

    Returns:
        pandas.DataFrame: A DataFrame with 'Country Name', 'Year', 'GDP per capita', and '% change'.
    """
    logging.info("Processing data ...")
    gdp_years = dfs[0].columns.values.tolist()
    population_years = dfs[1].columns.values.tolist()

    if len(gdp_years) != len(population_years):
        logging.warning(
            "GDP and Population do not have the same amount of years.")

    # Identify which years are identical for both dataframes
    columns_to_be_removed = []
    years = []
    for i in range(len(gdp_years)):
        try:
            year = int(gdp_years[i])
        except:
            continue

        if gdp_years[i] == population_years[i]:
            years.append(gdp_years[i])

    # DONE: Merge GDP and Population dataframes and calculate GDP per capita for each
    dfs[0].dropna(axis=1, how='all', inplace=True)
    dfs[1].dropna(axis=1, how='all', inplace=True)

    merged_df = pd.merge(dfs[0], dfs[1], on=[
                         'Country Name'], suffixes=('_GDP', '_Population'))

    for year in years:
        gdp_column = f'{year}_GDP'
        population_column = f'{year}_Population'
        columns_to_be_removed.append(gdp_column)
        columns_to_be_removed.append(population_column)

        merged_df[f'GDP per capita_{year}'] = merged_df[gdp_column] / \
            merged_df[population_column]

    # DONE: Calculate the % change of this value with respect to the previous year
    # defragmenting merged_df for efficiency reasons
    merged_df = merged_df.copy()

    for i in range(1, len(years)):
        current_year = years[i]
        previous_year = years[i - 1]
        merged_df[f'% change{current_year}'] = (merged_df[f'GDP per capita_{current_year}'] -
                                                merged_df[f'GDP per capita_{previous_year}']) / merged_df[f'GDP per capita_{previous_year}'] * 100

    merged_df.drop(columns=merged_df.filter(
        like='Indicator').columns.tolist()+columns_to_be_removed, inplace=True)
    merged_df.drop(columns=merged_df.filter(
        like='Country Code').columns.tolist(), inplace=True)

    # Melt the DataFrame for GDP per capita
    df_gdp = pd.melt(merged_df,
                     id_vars=['Country Name'],
                     value_vars=[
                         col for col in merged_df.columns if 'GDP per capita' in col],
                     var_name='Year',
                     value_name='GDP per capita')
    df_gdp['Year'] = df_gdp['Year'].str.extract('(\d{4})').astype(int)

    # Melt the DataFrame for % change
    df_change = pd.melt(merged_df,
                        id_vars=['Country Name'],
                        value_vars=[
                            col for col in merged_df.columns if '% change' in col],
                        var_name='Year',
                        value_name='% change')

    df_change['Year'] = df_change['Year'].str.extract('(\d{4})').astype(int)
    df_long = pd.merge(df_gdp, df_change, on=['Country Name', 'Year'])

    df_long = df_long[['Country Name', 'Year', 'GDP per capita', '% change']]
    df_long.reset_index(drop=True, inplace=True)
    df_long = df_long.sort_values(
        [df_long.columns[0], df_long.columns[1]], ascending=True)
    df_long = df_long.where(pd.notnull(df_long), "")

    # to produce the exact example of the description
    df_long.rename(columns={'Country Name': 'Text',
                   'Year': 'Value'}, inplace=True)

    logging.info("Processed data successfully!")
    return df_long


def create_dataset(df, api_token):
    """
    Creates a dataset on the remote server using the provided DataFrame.

    Args:
        df (pandas.DataFrame): The processed DataFrame to be uploaded.

    Returns:
        bool: True if the dataset was created successfully, False otherwise.
    """
    logging.info("Creating a dataset ...")
    create_dataset_api_endpoint = "https://app.23degrees.io/api/v2/content/dataset"

    # note: in production scenarios this would be located in a file or received through an environmental variable

    headers = {
        'accept': 'application/json',
        'Authorization': api_token,
        'Content-Type': 'application/json'
    }

    # tries to create a populated dataset
    payload = df.to_dict(orient='records')

    data = {
        "options": {
            "folderSlug": "empty dataset",
            "title": "GDP per capita example",
            "subTitle": "empty dataset",
            "description": "string",
            "source": "string",
            "updateInterval": "static",
            "access": "public",
            "numberLocal": "de",
            "numberShorten": True,
            "numberDecimalPlaces": 0,
            "preferredLayers": [
                "string"
            ]
        },
        "payload": payload
    }

    json_data = json.dumps(data)
    response = requests.post(create_dataset_api_endpoint,
                             headers=headers, data=json_data)
    if response.status_code == 200:
        logging.info("Dataset created successfully!")
    elif response.status_code == 400:
        logging.error("Error 400: Bad Request")
        logging.error("Error Details:", response.json())
    elif response.status_code == 401:
        logging.error("Error 401: Unauthorized. Check your API token.")
    elif response.status_code == 500:
        logging.error(
            "Error 500: Internal Server Error. Please try again later.")
    else:
        # Other errors: Handle any other status codes
        logging.error(
            f"Error {response.status_code}: An unexpected error occurred.")
        logging.error("Response Text:", response.text)
    if response.status_code != 200:
        return False
    return True


def main():
    """
    Main function to orchestrate the data pipeline process.

    This function performs the following steps:
    1. Configures logging settings.
    2. Loads user configuration
    3. Defines URLs for Population and GDP data from the World Bank (https://data.worldbank.org/)
    4. Creates the output directory if it doesn't exist.
    5. Downloads the Population and GDP datasets.
    6. Reads the datasets from the downloaded ZIP files.
    7. Processes the data to calculate GDP per capita and the percentage change year-over-year.
    8. Creates a dataset on a remote server using the processed data.

    The steps and error messages are logged in the CLI and in a file called "pipeline.log" in the directory of the execution. If any step encounters an error, the process stops.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting ...")
    config = load_config()

    population_url = config["DEFAULT"]["population_url"]
    gdp_url = config["DEFAULT"]["gdp_url"]
    output_path = config["DEFAULT"]["output_path"]
    api_token = config["DEFAULT"]["api_token"]

    population_output_path = os.path.join(output_path, "population.zip")
    gdp_output_path = os.path.join(output_path, "gdp.zip")

    try:
        Path(output_path).mkdir(parents=True, exist_ok=True)

        logging.info("Fetching Population data ...")
        download_url(population_url, population_output_path)

        logging.info("Fetching GDP data ...")
        download_url(gdp_url, gdp_output_path)

        dfs = read_data(output_path)
        merged_df = process_data(dfs)
        result = create_dataset(merged_df, api_token)

        if result:
            logging.info("Finished successfully!.")
        else:
            logging.error("Failed to create dataset.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.debug("Exception details", exc_info=True)


if __name__ == "__main__":
    main()
