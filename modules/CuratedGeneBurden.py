#!/usr/bin/env python
"""This module processes and returns target/disease evidence from curated gene burden data."""

import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from common.evidence import initialize_sparksession, write_evidence_strings_tsv

def process_gene_burden_curation(spark: SparkSession, curated_data: str):
    schema = StructType([
        StructField('projectId', StringType(), True),
        StructField('targetFromSource', StringType(), True),
        StructField('targetFromSourceId', StringType(), True),
        StructField('diseaseFromSource', StringType(), True),
        StructField('diseaseFromSourceMappedId', StringType(), True),
        StructField('resourceScore', DoubleType(), True),
        StructField('pValueMantissa', DoubleType(), True),
        StructField('pValueExponent', IntegerType(), True),
        StructField('oddsRatio', DoubleType(), True),
        StructField('ConfidenceIntervalLower', DoubleType(), True),
        StructField('ConfidenceIntervalUpper', DoubleType(), True),
        StructField('beta', DoubleType(), True),
        StructField('sex', StringType(), True),
        StructField('ancestry', StringType(), True),
        StructField('ancestryId', StringType(), True),
        StructField('cohortId', StringType(), True),
        StructField('studySampleSize', IntegerType(), True),
        StructField('studyCases', IntegerType(), True),
        StructField('studyCasesWithQualifyingVariants', IntegerType(), True),
        StructField('allelicRequirements', StringType(), True),
        StructField('studyId', StringType(), True),
        StructField('statisticalMethod', StringType(), True),
        StructField('statisticalMethodOverview', StringType(), True),
        StructField('literature', StringType(), True),
        StructField('url', StringType(), True),
    ])
    
    curated_df = spark.read.csv(curated_data, sep='\t', header=True, schema=schema)
    processed_df = (
        curated_df
        .withColumn('oddsRatioConfidenceIntervalLower', f.when(f.col('oddsRatio').isNotNull(), f.col('ConfidenceIntervalLower')))
        .withColumn('oddsRatioConfidenceIntervalUpper', f.when(f.col('oddsRatio').isNotNull(), f.col('ConfidenceIntervalUpper')))
        .withColumn('betaConfidenceIntervalLower', f.when(f.col('beta').isNotNull(), f.col('ConfidenceIntervalLower')))
        .withColumn('betaConfidenceIntervalUpper', f.when(f.col('beta').isNotNull(), f.col('ConfidenceIntervalUpper')))
        .drop('ConfidenceIntervalLower', 'ConfidenceIntervalUpper')
        .withColumn('literature', f.array(f.col('literature')))
        .withColumn('allelicRequirements', f.when(f.col('allelicRequirements').isNotNull(), f.array(f.col('allelicRequirements'))))
        .withColumn('sex', f.split(f.col('sex'), ', '))
        .withColumn('datasourceId', f.lit('gene_burden'))
        .withColumn('datatypeId', f.lit('genetic_association'))
        .drop('targetFromSource')
    )

    logging.info(f'Processed {processed_df.count()} records from curated data.')
    return processed_df

def main(spark: SparkSession, curated_data: str, output_file: str = None):
    evd_df = process_gene_burden_curation(spark, curated_data)
    if output_file:
        write_evidence_strings_tsv(evd_df, output_file)
        logging.info(f"Evidence strings have been saved to {output_file}.")
    return evd_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--curated_data',
        help='Input TSV file containing the curated gene burden evidence.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output',
        help='Optional: Output TSV file for the processed gene burden evidence.',
        type=str,
    )
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    spark = initialize_sparksession()
    main(spark, args.curated_data, args.output)