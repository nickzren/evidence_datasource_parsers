#!/usr/bin/env python
"""This module extracts and processes target/disease evidence from the Finngen data."""

import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from common.evidence import write_evidence_strings_tsv, import_trait_mappings, initialize_sparksession

def process_finngen_gene_burden(spark: SparkSession, finngen_data: str, finngen_manifest: str):
    """Process Finngen's loss of function burden results."""
    
    # Load the manifest and Finngen data
    manifest = spark.read.json(finngen_manifest).selectExpr("phenocode as PHENO", "phenostring as diseaseFromSource")
    
    finngen_df = (
        spark.read.csv(finngen_data, header=True, sep="\t")
        # Join description of Finngen's endpoint from the manifest
        .join(manifest, "PHENO", "left")
        # Bring in mappings for disease IDs
        .join(
            import_trait_mappings(spark),
            on="diseaseFromSource",
            how="left",
        )
        .withColumn("resourceScore", 10**-F.col("LOG10P").cast(DoubleType()))
        .withColumn("pValueExponent", F.log10(F.col("resourceScore")).cast("int") - F.lit(1))
        .withColumn("pValueMantissa", F.round(F.col("resourceScore") / F.pow(F.lit(10), F.col("pValueExponent")), 3))
        .select(
            F.lit("gene_burden").alias("datasourceId"),
            F.lit("finnish").alias("ancestry"),
            F.lit("HANCESTRO_0321").alias("ancestryId"),
            F.col("BETA").alias("beta").cast("float"),
            (F.col("BETA") - F.col("SE")).alias("betaConfidenceIntervalLower").cast("float"),
            (F.col("BETA") + F.col("SE")).alias("betaConfidenceIntervalUpper").cast("float"),
            F.lit("FinnGen R11").alias("cohortId"),
            F.lit("genetic_association").alias("datatypeId"),
            F.col("diseaseFromSource"),
            F.col("PHENO").alias("diseaseFromSourceId"),
            F.col("diseaseFromSourceMappedId"),
            F.lit("FinnGen").alias("projectId"),
            F.col("resourceScore"),
            F.col("pValueExponent"),
            F.col("pValueMantissa"),
            F.lit("R11").alias("releaseVersion"),
            F.lit("LoF burden").alias("statisticalMethod"),
            F.lit("Burden test carried out with LoF variants with MAF smaller than 1%.").alias("statisticalMethodOverview"),
            F.lit(453733).alias("studySampleSize"),
            F.split(F.col("ID"), "\.")[0].alias("targetFromSourceId"),
        )
    )

    gene_count = finngen_df.select("targetFromSourceId").distinct().count()
    statistical_significance = 0.05 / gene_count  # Applying a Bonferroni correction
    return finngen_df.filter(F.col("resourceScore") <= statistical_significance).distinct()


def main(spark: SparkSession, finngen_data: str, finngen_manifest: str, output_file: str = None):
    """Main function to process and optionally save Finngen gene burden evidence."""
    logging.info("Starting processing for Finngen gene burden data.")
    evd_df = process_finngen_gene_burden(spark, finngen_data, finngen_manifest)
    logging.info(f"Processed {evd_df.count()} records from Finngen data.")

    if output_file:
        write_evidence_strings_tsv(evd_df, output_file)
        logging.info(f"Evidence strings have been saved to {output_file}.")

    return evd_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finngen_data",
        help="Input tab-delimited file containing Finngen's burden tests.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--finngen_manifest",
        help="Input JSON file with Finngen's traits manifest.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Optional: Output TSV file following the gene_burden evidence data model.",
        type=str,
    )

    args = parser.parse_args()

    # Initialize Spark session
    spark = initialize_sparksession()
    main(spark, args.finngen_data, args.finngen_manifest, args.output)