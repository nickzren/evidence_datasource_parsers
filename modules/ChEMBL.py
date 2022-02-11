#!/usr/bin/env python
"""This module adds the category of why a clinical trial has stopped early to the ChEMBL evidence."""

import argparse
import logging
import sys

import pyspark.sql.functions as F
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import ArrayType, StringType

from common.evidence import detect_spark_memory_limit, write_evidence_strings



def main(chembl_evidence: str, predictions: str, output_file:str) -> None:
    """
    This module adds the studyStopReasonCategories to the ChEMBL evidence as a result of the categorisation of the clinical trial reason to stop.
    Args:
        chembl_evidence: Input gzipped JSON with the evidence submitted by ChEMBL. 
        predictions: Input TSV containing the categories of the clinical trial reason to stop. Instructions for applying the ML model here: https://github.com/ireneisdoomed/stopReasons.
        output_file: Output gzipped json file containing the ChEMBL evidence with the additional studyStopReasonCategories field.
        log_file: Destination of the logs generated by this script. Defaults to None.
    """
    logging.info(f'ChEMBL evidence JSON file: {chembl_evidence}')
    logging.info(f'Classes of reason to stop table: {predictions}')

    # Load input into dataframes
    chembl_df = spark.read.json(chembl_evidence)
    predictions_df = (
        load_stop_reasons_classes(predictions)
        
    )



    # Join datasets
    evd_df = (
        chembl_df.join(predictions_df, chembl_df['studyStopReason'] == predictions_df['why_stopped'], how='left')
        .withColumnRenamed('subclasses', 'studyStopReasonCategories')
        .drop('why_stopped', 'superclasses')
        .distinct()
    )

    # We expect that ~10% of evidence strings have a reason to stop assigned
    # It is asserted that this fraction is between 9 and 11% of the total count
    total_count = evd_df.count()
    early_stopped_count = evd_df.filter(F.col('studyStopReasonCategories').isNotNull()).count()

    if not 0.08 < early_stopped_count / total_count < 0.11:
        raise AssertionError (f'The fraction of evidence with a CT reason to stop class is not as expected ({early_stopped_count / total_count}).')

    # Write output
    logging.info('Evidence strings have been processed. Saving...')
    write_evidence_strings(evd_df, output_file)

    total_count = evd_df.count()
    early_stopped_count = evd_df.filter(F.col('studyStopReasonCategories').isNull())
    logging.info(
        f'{total_count} evidence strings have been saved to {output_file}. Exiting.'
    )

def load_stop_reasons_classes(predictions: str) -> DataFrame:
    """
    Loads TSV file with predictions of the reasons to stop and their assigned category.
    List of categories must be converted to an array format and formatted with a nice name.
    """

    CATEGORIESMAPPINGS   = {
    'Business_Administrative': 'Business or administrative',
    'Logistics_Resources': 'Logistics or resources',
    'Covid19': 'COVID-19',
    'Safety_Sideeffects': 'Safety or side effects',
    'Endpoint_Met': 'Met endpoint',
    'Insufficient_Enrollment': 'Insufficient enrollment',
    'Negative': 'Negative',
    'Study_Design': 'Study design',
    'Invalid_Reason': 'Invalid reason',
    'Study_Staff_Moved': 'Study staff moved',
    'Another_Study': 'Another study',
    'No_Context': 'No context',
    'Regulatory': 'Regulatory',
    'Interim_Analysis': 'Interim analysis',
    'Success': 'Success',
    'Ethical_Reason': 'Ethical reason',
    'Insufficient_Data': 'Insufficient data',
    }

    schema = ArrayType(StringType())
    sub_mapping_col = F.map_from_entries(F.array(*[F.struct(F.lit(k), F.lit(v)) for k, v in CATEGORIESMAPPINGS.items()]))

    return (
        spark.read.csv(predictions, sep='\t', header=True)

        # Lists are represented as strings. They must be converted
        .withColumn("subclasses", F.from_json(F.regexp_replace(F.col('subclasses'), "(u')", "'"), schema=schema))
        .withColumn("superclasses", F.from_json(F.regexp_replace(F.col('superclasses'), "(u')", "'"), schema=schema))
        
        # Create a MapType column to convert each element of the subclasses array
        .withColumn("subMappings", sub_mapping_col)
        .withColumn('subclasses', F.expr('transform(subclasses, x -> element_at(subMappings, x))'))
        .drop('subMappings')
    )

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

def get_parser():
    """Get parser object for script ChEMBL.py."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--chembl_evidence',
        help='Input gzipped JSON with the evidence submitted by ChEMBL',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--predictions',
        help='Input TSV containing the categories of the clinical trial reason to stop. Instructions for applying the ML model here: https://github.com/ireneisdoomed/stopReasons.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output', help='Output gzipped json file following the target safety liabilities data model.', type=str, required=True
    )
    parser.add_argument(
        '--log_file',
        help='Destination of the logs generated by this script. Defaults to None',
        type=str,
        nargs='?',
        default=None
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # Logger initializer. If no log_file is specified, logs are written to stderr
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if args.log_file:
        logging.config.fileConfig(filename=args.log_file)
    else:
        logging.StreamHandler(sys.stderr)

    global spark
    spark = initialize_spark()

    main(
        chembl_evidence=args.chembl_evidence,
        predictions=args.predictions,
        output_file=args.output
    )