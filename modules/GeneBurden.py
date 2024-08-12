#!/usr/bin/env python
"""This module brings together and exports target/disease evidence generated by AzGeneBurden.py and RegeneronGeneBurdeb.py."""

import argparse
from functools import partial, reduce
import logging
import sys

from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as f
import pyspark.sql.types as T

from common.evidence import apply_bonferroni_correction, import_trait_mappings, initialize_sparksession, write_evidence_strings
from AzGeneBurden import main as process_az_gene_burden
from GenebassGeneBurden import main as process_genebass_gene_burden


def main(
    spark: SparkSession,
    az_binary_data: str,
    az_quant_data: str,
    az_genes_links: str,
    az_phenotypes_links: str,
    curated_data: str,
    genebass_data: str,
    finngen_data: str,
    finngen_manifest: str,
) -> DataFrame:
    """This module brings together and exports target/disease evidence generated by AzGeneBurden.py and RegeneronGeneBurdeb.py."""

    burden_evidence_sets = [
        # Generate evidence from AZ data:
        process_az_gene_burden(spark, az_binary_data, az_quant_data, az_genes_links, az_phenotypes_links).persist(),
        # Generate evidence from manual data:
        process_gene_burden_curation(spark, curated_data),
        # Generate evidence from GeneBass data:
        process_genebass_gene_burden(spark, genebass_data),
        # Generate evidence from Finngen data:
        process_finngen_gene_burden(spark, finngen_data, finngen_manifest)
    ]

    unionByDiffSchema = partial(DataFrame.unionByName, allowMissingColumns=True)
    evd_df = reduce(unionByDiffSchema, burden_evidence_sets).distinct()
    logging.info(f'Total number of gene_burden evidence: {evd_df.count()}')

    return evd_df

def process_finngen_gene_burden(spark: SparkSession, finngen_data: str, finngen_manifest: str,) -> DataFrame:
    """Process Finngen's loss of function burden results."""

    manifest = spark.read.json(finngen_manifest).selectExpr("phenocode as PHENO", "phenostring as diseaseFromSource")
    
    finngen_df = (
        spark.read.csv(finngen_data, header=True, sep="\t")
        # Bring description of Finngen's endpoint from manifest
        .join(manifest, "PHENO", "left")
        # Bring disease to ID LUT
        .join(
            import_trait_mappings(spark),
            on="diseaseFromSource",
            how="left",
        )
        .withColumn(
            "resourceScore",
            10**-f.col("LOG10P")
        )
        .withColumn(
            "pValueExponent",
            f.log10(f.col("resourceScore")).cast("int") - f.lit(1)
        )
        .withColumn(
            "pValueMantissa",
            f.round(
                f.col("resourceScore") / f.pow(f.lit(10), f.col("pValueExponent")), 3
            )
        )
        .select(
            f.lit("finnish").alias("ancestry"),
            f.lit("HANCESTRO_0321").alias("ancestryId"),
            f.col("BETA").alias("beta"),
            (f.col("BETA") - f.col("SE")).alias("betaConfidenceIntervalLower"),
            (f.col("BETA") + f.col("SE")).alias("betaConfidenceIntervalUpper"),
            f.lit("FinnGen R11").alias("cohortId"),
            f.lit("genetic_association").alias("datatypeId"),
            f.col("diseaseFromSource"),
            f.col("PHENO").alias("diseaseFromSourceId"),
            f.col("diseaseFromSourceMappedId"),
            f.lit("FinnGen").alias("projectId"),
            f.col("resourceScore"),
            f.col("pValueExponent"),
            f.col("pValueMantissa"),
            f.lit("R11").alias("releaseVersion"),
            f.lit("LoF burden").alias("statisticalMethod"),
            f.lit("Burden test carried out with LoF variants with MAF smaller than 1%.").alias("statisticalMethodOverview"),
            f.lit(453733).alias("studySampleSize"),
            f.split(f.col("ID"), "\.")[0].alias("targetFromSourceId"),
        )
    )
    gene_count = finngen_df.select("targetFromSourceId").distinct().count()
    statistical_significance = apply_bonferroni_correction(gene_count)
    return (
        finngen_df
        .filter(f.col("resourceScore") <= statistical_significance)
        .distinct()
    )
        

def process_gene_burden_curation(spark: SparkSession, curated_data: str) -> DataFrame:
    """Process manual gene burden evidence."""

    logging.info(f'File with the curated burden associations: {curated_data}')
    manual_df = read_gene_burden_curation(spark, curated_data)
    logging.info(f'Total number of imported manual gene_burden evidence: {manual_df.count()}')

    manual_df = (
        manual_df
        # The columns practically follow the schema, only small things need to be parsed
        # 1. Confidence intervals are detached
        .withColumn(
            'oddsRatioConfidenceIntervalLower', f.when(f.col('oddsRatio').isNotNull(), f.col('ConfidenceIntervalLower'))
        )
        .withColumn(
            'oddsRatioConfidenceIntervalUpper', f.when(f.col('oddsRatio').isNotNull(), f.col('ConfidenceIntervalUpper'))
        )
        .withColumn('betaConfidenceIntervalLower', f.when(f.col('beta').isNotNull(), f.col('ConfidenceIntervalLower')))
        .withColumn('betaConfidenceIntervalUpper', f.when(f.col('beta').isNotNull(), f.col('ConfidenceIntervalUpper')))
        .drop('ConfidenceIntervalLower', 'ConfidenceIntervalUpper')
        # 2. Collect PMID and allelic requirements in an array
        .withColumn('literature', f.array(f.col('literature')))
        .withColumn(
            'allelicRequirements',
            f.when(f.col('allelicRequirements').isNotNull(), f.array(f.col('allelicRequirements'))),
        )
        # 3. Split the sex column to form an array
        .withColumn('sex', f.split(f.col('sex'), ', '))
        # 4. Add hardcoded values and drop URLs (will be handled by the FE) and HGNC symbols
        .withColumn('datasourceId', f.lit('gene_burden'))
        .withColumn('datatypeId', f.lit('genetic_association'))
        .drop('url', 'targetFromSource', 'studyId')
        .distinct()
    )

    return manual_df


def read_gene_burden_curation(spark: SparkSession, curated_data: str) -> DataFrame:
    """Read manual gene burden curation from remote to a Spark DataFrame."""

    schema = T.StructType(
        [
            T.StructField('projectId', T.StringType(), True),
            T.StructField('targetFromSource', T.StringType(), True),
            T.StructField('targetFromSourceId', T.StringType(), True),
            T.StructField('diseaseFromSource', T.StringType(), True),
            T.StructField('diseaseFromSourceMappedId', T.StringType(), True),
            T.StructField('resourceScore', T.DoubleType(), True),
            T.StructField('pValueMantissa', T.DoubleType(), True),
            T.StructField('pValueExponent', T.IntegerType(), True),
            T.StructField('oddsRatio', T.DoubleType(), True),
            T.StructField('ConfidenceIntervalLower', T.DoubleType(), True),
            T.StructField('ConfidenceIntervalUpper', T.DoubleType(), True),
            T.StructField('beta', T.DoubleType(), True),
            T.StructField('sex', T.StringType(), True),
            T.StructField('ancestry', T.StringType(), True),
            T.StructField('ancestryId', T.StringType(), True),
            T.StructField('cohortId', T.StringType(), True),
            T.StructField('studySampleSize', T.IntegerType(), True),
            T.StructField('studyCases', T.IntegerType(), True),
            T.StructField('studyCasesWithQualifyingVariants', T.IntegerType(), True),
            T.StructField('allelicRequirements', T.StringType(), True),
            T.StructField('studyId', T.StringType(), True),
            T.StructField('statisticalMethod', T.StringType(), True),
            T.StructField('statisticalMethodOverview', T.StringType(), True),
            T.StructField('literature', T.StringType(), True),
            T.StructField('url', T.StringType(), True),
        ]
    )
    spark.sparkContext.addFile(curated_data)
    return spark.read.csv(
        SparkFiles.get(curated_data.split('/')[-1]), sep='\t', header=True, schema=schema
    )


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
        "--az_genes_links",
        help="Input CSV file that consists of a look up table between a gene and its link in the AZ Phewas Portal.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--az_phenotypes_links",
        help="Input CSV file that consists of a look up table between a phenotype and its link in the AZ Phewas Portal.",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--curated_data',
        help='Input remote CSV file containing the gene burden manual curation.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--genebass_data',
        help='Input parquet files with Genebass\'s burden associations.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--finngen_data',
        help='Input tab delimited table containing all Finngen\'s burden tests. Downloaded from their public bucket in GCS.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--finngen_manifest',
        help='Input JSON with Finngen\'s traits manifest. Necessary to extract the trait endpoint based on the trait description, which is what the curation table contains.',
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
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if args.log_file:
        logging.config.fileConfig(filename=args.log_file)
    else:
        logging.StreamHandler(sys.stderr)

    spark = initialize_sparksession()

    evd_df = main(
        spark=spark,
        az_binary_data=args.az_binary_data,
        az_quant_data=args.az_quant_data,
        az_genes_links=args.az_genes_links,
        az_phenotypes_links=args.az_phenotypes_links,
        curated_data=args.curated_data,
        genebass_data=args.genebass_data,
        finngen_data=args.finngen_data,
        finngen_manifest=args.finngen_manifest,
    )

    write_evidence_strings(evd_df, args.output)
    logging.info(f'Evidence strings have been saved to {args.output}. Exiting.')
