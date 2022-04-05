#!/usr/bin/env python
"""This module brings together and exports target/disease evidence generated by AzGeneBurden.py and RegeneronGeneBurdeb.py."""

import argparse
import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from common.evidence import initialize_sparksession, write_evidence_strings
from RegeneronGeneBurden import main as process_regeneron_gene_burden
from AzGeneBurden import main as process_az_gene_burden


def main(
    az_binary_data: str, az_quant_data: str, regeneron_data: str, gwas_studies: str, spark_instance: SparkSession
) -> DataFrame:
    """This module brings together and exports target/disease evidence generated by AzGeneBurden.py and RegeneronGeneBurdeb.py."""

    regeneron_evd = process_regeneron_gene_burden(regeneron_data, gwas_studies, spark_instance=spark_instance)
    az_evd = process_az_gene_burden(az_binary_data, az_quant_data, spark_instance=spark_instance)

    evd_df = regeneron_evd.unionByName(az_evd, allowMissingColumns=True).distinct()
    logging.info(f"Total number of gene_burden evidence: {evd_df.count()}")

    return evd_df


def get_parser():
    """Get parser object for script GeneBurden.py."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--az_binary_data',
        help='Input parquet files with AZ\'s PheWAS associations of binary traits.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--az_quant_data',
        help='Input parquet files with AZ\'s PheWAS associations of quantitative traits.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--regeneron_data',
        help='Input Excel file with the data published in PMID:34662886.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--gwas_studies',
        help='Input TSV containing all GWAS Catalog studies.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output',
        help='Output gzipped json file following the gene_burden evidence data model.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--log_file',
        help='Destination of the logs generated by this script. Defaults to None',
        type=str,
        nargs='?',
        default=None,
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Logger initializer. If no log_file is specified, logs are written to stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.log_file:
        logging.config.fileConfig(filename=args.log_file)
    else:
        logging.StreamHandler(sys.stderr)

    spark = initialize_sparksession()

    evd_df = main(
        az_binary_data=args.az_binary_data,
        az_quant_data=args.az_quant_data,
        regeneron_data=args.regeneron_data,
        gwas_studies=args.gwas_studies,
        spark_instance=spark,
    )

    write_evidence_strings(evd_df, args.output)
    logging.info(f"Evidence strings have been saved to {args.output}. Exiting.")
