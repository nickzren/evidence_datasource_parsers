#!/usr/bin/env python
"""This module puts together data from different sources that describe target safety liabilities."""

import logging
import sys

import fire
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.sql.functions import (
    col,
    lit,
    udf,
    when,
    explode_outer,
    substring,
    array,
    regexp_extract,
    concat_ws,
)

from common.evidence import detect_spark_memory_limit


def main(adverse_events: str, safety_risk: str, toxcast: str, output:str, log_file:str):
    """
    This module puts together data from different sources that describe target safety liabilities.

    Args:
        adverse_events: Input table containing adverse events associated with targets that have been collected from relevant publications.
        safety_risk: Input table containing cardiovascular safety liabilities associated with targets that have been collected from relevant publications.
        toxcast: Input table containing biological processes associated with relevant targets that have been observed in toxicity assays.
        output: Output gzipped json file following the target safety liabilities data model.
        log_file: Destination of the logs generated by this script.
    """

    global spark
    spark = initialize_spark()
    initialize_logger(log_file)

    # Load and process the input files into dataframes
    adverse_events_df = process_adverse_events(adverse_events)
    safety_risk_df = process_safety_risk(safety_risk)
    toxcast_df = process_toxcast(toxcast)

    # Write output
    logging.info('Evidence strings have been processed. Saving...')
    print('Output')
    pass

def process_adverse_events(adverse_events: str) -> DataFrame:
    """Loads and processes the input table containing adverse events associated with targets that have been collected from relevant publications."""

    pass

def process_safety_risk(safety_risk: str) -> DataFrame:
    """Loads and processes the input table containing cardiovascular safety liabilities associated with targets that have been collected from relevant publications."""

    pass

def process_toxcast(toxcast: str) -> DataFrame:
    """Loads and processes the ToxCast input table containing biological processes associated with relevant targets that have been observed in toxicity assays."""

    pass

def initialize_spark():
    """Spins up a Spark session."""

    # Initialize spark session
    spark_mem_limit = detect_spark_memory_limit()
    spark_conf = (
        SparkConf()
        .set('spark.driver.memory', f'{spark_mem_limit}g')
        .set('spark.executor.memory', f'{spark_mem_limit}g')
        .set('spark.driver.maxResultSize', '0')
        .set('spark.debug.maxToStringFields', '2000')
        .set('spark.sql.execution.arrow.maxRecordsPerBatch', '500000')
    )
    spark = (
        SparkSession.builder
        .config(conf=spark_conf)
        .master('local[*]')
        .getOrCreate()
    )
    logging.info(f'Spark version: {spark.version}')

    return spark

def initialize_logger(logFile=None):
    """Logger initializer. If no logfile is specified, logs are written to stderr."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if logFile:
        logging.config.fileConfig(filename=logFile)
    else:
        logging.StreamHandler(sys.stderr)


if __name__ == '__main__':
    fire.Fire(main)