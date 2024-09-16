# Population Data Pipeline for GDP and Population Data Processing

This project is a data pipeline that downloads and processes GDP and population data from the World Bank API.
Furthermore, it calculates GDP per capita, and computes its yearly percentage change.
The processed data is later uploaded to a remote server to create a dataset via an API endpoint so a graph can be created using the interface of the website: app.23degrees.io

## Features

- Downloads GDP and Population data in CSV format from the World Bank.
- Processes the data to calculate GDP per capita and its percentage change year-over-year.
- Uses a configuration file (`config.ini`) for user-configurable options like URLs and API tokens.
- Uploads the processed data to a remote server using an API.
- Logs pipeline steps to both the console and a log file (`pipeline.log`).

## Requirements

- Tested with Python 3.11.9 64-bit
- The api_token needs to be inserted after making an account using the following link: https://app.23degrees.io/user/account

# You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

- Run the pipeline
```bash
python gdp_population_data_pipeline.py
```

## Expected Results
Local execution example
![The picture "local_execution.png" was not loaded, please download separately](https://github.com/ErmisCho/population-data-pipeline/blob/main/github_pictures/local_execution.png)

Result - Website table: the data itself on the website of 23degrees.io
![The picture "website_table.png" was not loaded, please download separately](https://github.com/ErmisCho/population-data-pipeline/blob/main/github_pictures/website_table.png)

Result - Final Result: a potential user-friendly demonostration of the processed data
![The picture "final_result.png" was not loaded, please download separately](https://github.com/ErmisCho/population-data-pipeline/blob/main/github_pictures/final_result.png)

## Logs
Example of how the logs file "pipeline.log" is structured
![The picture "pipeline_example.png" was not loaded, please download separately](https://github.com/ErmisCho/population-data-pipeline/blob/main/github_pictures/pipeline_example.png)