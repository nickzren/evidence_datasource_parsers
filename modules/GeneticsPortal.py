#!/usr/bin/env python
"This script pulls together data from Open Targets Genetics portal to generate disease/target evidence strings for the Platform."

import argparse
import sys
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
import logging


def load_eco_dict(inf):
    """
    Loads the csq to eco scores into a dict
    Returns: dict
    """

    # Load
    eco_df = spark.read.csv(inf, sep="\t", header=True, inferSchema=True).select(
        "Term", "Accession", col("eco_score").cast(DoubleType())
    )

    # Convert to python dict
    eco_dict = {}
    eco_link_dict = {}
    for row in eco_df.collect():
        eco_dict[row.Term] = row.eco_score
        eco_link_dict[row.Term] = row.Accession

    return (eco_dict, eco_link_dict)

def main(
    locus2gene: str,
    toploci: str,
    study_index: str,
    variant_index: str,
    vep_consequences: str,
    threshold: float,
    output_file: str

):
    logging.info(f"Locus2gene table: {locus2gene}")
    logging.info(f"Top locus table: {toploci}")
    logging.info(f"Study table: {study_index}")
    logging.info(f"Variant index table: {variant_index}")
    logging.info(f"ECO code table: {vep_consequences}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"l2g score threshold: {threshold}")

    # Load and process the input files into dataframes
    l2g_df = process_l2g_table(locus2gene, threshold)
    pvals_df = process_toploci_table(toploci)
    studies_df = process_study_table(study_index)


    # Get mapping for rsIDs:
    variant_info = (
        spark.read.parquet(variant_index)
        # chrom_b38|pos_b38
        # Explode consequences, only keeping canonical transcript
        .selectExpr(
            "chrom_b38 as chrom",
            "pos_b38 as pos",
            "ref",
            "alt",
            "rsid"
        )
    )

    # Get most severe consequences:

    # Load term to eco score dict
    # (eco_dict,eco_link_dict) = spark.sparkContext.broadcast(load_eco_dict(in_csq_eco))
    eco_dicts = spark.sparkContext.broadcast(load_eco_dict(vep_consequences))

    get_link = udf(lambda x: eco_dicts.value[1][x], StringType())

    # Extract most sereve csq per gene.
    # Create UDF that reverse sorts csq terms using eco score dict, then select
    # the first item. Then apply UDF to all rows in the data.
    get_most_severe = udf(
        lambda arr: sorted(
            arr, key=lambda x: eco_dicts.value[0].get(x, 0), reverse=True
        )[0],
        StringType(),
    )

    variant_consequences = (
        spark.read.parquet(variant_index)

        # Explode consequences, only keeping canonical transcript
        .selectExpr(
            "chrom_b38 as chrom",
            "pos_b38 as pos",
            "ref",
            "alt",
            "vep.most_severe_consequence as most_severe_csq",
            """explode(
                filter(vep.transcript_consequences, x -> x.canonical == 1)
            ) as tc
            """,
        )
        # Keep required fields from consequences struct
        .selectExpr(
            "chrom",
            "pos",
            "ref",
            "alt",
            "most_severe_csq",
            "tc.gene_id as gene_id",
            "tc.consequence_terms as csq_arr",
        )
        
        # Get most severe consequences
        .withColumn("most_severe_gene_csq", get_most_severe(col("csq_arr")))
        .withColumn("consequence_link", get_link(col("most_severe_gene_csq")))
    )

    # Join datasets together
    processed = (
        l2g_df

        # Join L2G to pvals, using study and variant info as key
        .join(pvals_df, on=["study_id", "chrom", "pos", "ref", "alt"])

        # Join this to the study info, using study_id as key
        .join(studies_df, on="study_id", how="inner")

        # Join transcript consequences
        .join(
            variant_consequences, on=["chrom", "pos", "ref", "alt", "gene_id"], how="left"
        )
        # Bring rsIDs
        .join(variant_info, on=["chrom", "pos", "ref", "alt"], how="left")

        # Filling missing consequences
        .fillna(
            {
                "most_severe_gene_csq": "intergenic_variant",
                "consequence_link": "http://purl.obolibrary.org/obo/SO_0001628",
            }
        )
    )

    # Write output
    logging.info("Generating evidence:")
    (
        processed.withColumn(
            "literature",
            when(
                col("pmid") != "", array(regexp_extract(col("pmid"), r"PMID:(\d+)$", 1))
            ).otherwise(None),
        )
        .select(
            lit("ot_genetics_portal").alias("datasourceId"),
            lit("genetic_association").alias("datatypeId"),
            col("gene_id").alias("targetFromSourceId"),
            col("efo").alias("diseaseFromSourceMappedId"),
            col("literature"),
            col("pub_author").alias("publicationFirstAuthor"),
            "projectId",
            substring(col("pub_date"), 1, 4).cast(IntegerType()).alias("publicationYear"),
            col("trait_reported").alias("diseaseFromSource"),
            col("study_id").alias("studyId"),
            col("sample_size").alias("studySampleSize"),
            col("pval_mantissa").alias("pValueMantissa"),
            col("pval_exponent").alias("pValueExponent"),
            col("odds_ratio").alias("oddsRatio"),
            col("oddsr_ci_lower").alias("oddsRatioConfidenceIntervalLower"),
            col("oddsr_ci_upper").alias("oddsRatioConfidenceIntervalUpper"),
            col("beta").alias("beta"),
            col("beta_ci_lower").alias("betaConfidenceIntervalLower"),
            col("beta_ci_upper").alias("betaConfidenceIntervalUpper"),
            col("y_proba_full_model").alias("resourceScore"),
            col("rsid").alias("variantRsId"),
            concat_ws("_", col("chrom"), col("pos"), col("ref"), col("alt")).alias(
                "variantId"
            ),
            regexp_extract(col("consequence_link"), r"\/(SO.+)$", 1).alias(
                "variantFunctionalConsequenceId"
            ),
        )
        .dropDuplicates(
            ["variantId", "studyId", "targetFromSourceId", "diseaseFromSourceMappedId"]
        )
        .coalesce(1)
        .write.format("json")
        .mode("overwrite")
        .option("compression", "gzip")
        .save(output_file)
    )
    logging.info(f'Genetics evidence strings have been saved to {output_file}. Exiting.')

    return 0

def get_parser():
    """Get parser object for script .py."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--locus2gene",
        help="Input table containing locus to gene scores.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--toploci",
        help="Table containing top loci for all studies.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--study", help="Table with all the studies.", type=str, required=True
    )
    parser.add_argument(
        "--variantIndex",
        help="Table with the variant indices (from gnomad 2.x).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ecoCodes", help="Table with consequence ECO codes.", type=str, required=True
    )
    parser.add_argument(
        "--outputFile", help="Output gzipped json file.", type=str, required=True
    )
    parser.add_argument(
        "--threshold",
        help="Threshold applied on l2g score for filtering.",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--logFile",
        help="Destination of the logs generated by this script.",
        type=str,
        required=False,
    )

    return parser

def initialize_logger(logFile=None):
    """Logger initializer. If no logfile is specified, logs are written to stderr."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if logFile:
        logging.config.fileConfig(filename=logFile)
    else:
        logging.StreamHandler(sys.stderr)

def initialize_spark():
    """Spins up a Spark session."""

    sparkConf = (
        SparkConf()
        .set("spark.driver.memory", "15g")
        .set("spark.executor.memory", "15g")
        .set("spark.driver.maxResultSize", "0")
        .set("spark.debug.maxToStringFields", "2000")
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "500000")
    )
    spark = SparkSession.builder.config(conf=sparkConf).master("local[*]").getOrCreate()
    logging.info(f"Spark version: {spark.version}")

    return spark

def process_l2g_table(
    locus2gene: str,
    threshold: float
) -> DataFrame:
    """Loads and processes locus-to-gene (L2G) score data."""

    return (
        spark.read.parquet(locus2gene)

        # Keep results trained on high or medium confidence gold-standards
        .filter(col("training_gs") == "high_medium")
        # Keep results from xgboost model
        .filter(col("training_clf") == "xgboost")
        # Keep rows with l2g score above the threshold:
        .filter(col("y_proba_full_model") >= threshold)

        # Only keep study, variant, gene and score info
        .select(
            "study_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene_id",
            "y_proba_full_model",
        )
    )

def process_study_table(study_index:str) -> DataFrame:
    """Load disease information from the study table."""

    return (
        spark.read.json(study_index)
        .select(
            "study_id",
            "pmid",
            "pub_date",
            "pub_author",
            "trait_reported",
            "trait_efos",
            col("n_initial").alias("sample_size")
        )

        # Assign project based on the study author information
        .withColumn(
            "projectId",
            when(col("study_id").contains("FINNGEN"), "FINNGEN")
            .when(col("study_id").contains("NEALE"), "NEALE")
            .when(col("study_id").contains("SAIGE"), "SAIGE")
            .when(col("study_id").contains("GCST"), "GCST"),
        )

        # Warning! Not all studies have an EFO annotated (trait_efos is an empty array)
        # Also, some have multiple EFOs!
        # Studies with no EFO are kept, the array is exploded to capture each mapped trait
        .withColumn("efo", explode_outer(col("trait_efos")))
        .drop("trait_efos")

        # Drop records with HANCESTRO IDs as mapped trait
        .filter((~col("efo").contains('HANCESTRO')) | (col('efo').isNull()))
    )

def process_toploci_table(toploci:str) -> DataFrame:
    """Load association statistics (only pvalue is required) from top loci table."""
    
    return (
        spark.read.parquet(toploci)
        .select(
            "study_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "beta",
            "beta_ci_lower",
            "beta_ci_upper",
            "pval_mantissa",
            "pval_exponent",
            "odds_ratio",
            "oddsr_ci_lower",
            "oddsr_ci_upper",
        )

        # Problem: Large OR values which cannot be represented with a single precision float are
        # automatically casted to "infinity" when the df is exported to JSON, hence failing validation.
        # This was also a problem for ES, as these evidence were not being loaded (https://github.com/opentargets/platform/issues/1687)
        # Decision: OR is set to null.
        .filter((col('odds_ratio') < 2**62) | (col('odds_ratio').isNull()))
        .filter((col('oddsr_ci_lower') < 2**62) | (col('oddsr_ci_lower').isNull()))
        .filter((col('oddsr_ci_upper') < 2**62) | (col('oddsr_ci_upper').isNull()))
    )

if __name__ == "__main__":
    args = get_parser().parse_args()
    initialize_logger(args.logFile)

    global spark
    spark = initialize_spark()

    main(
        locus2gene=args.locus2gene,
        toploci=args.toploci,
        study_index=args.study,
        variant_index=args.variantIndex,
        vep_consequences=args.ecoCodes,
        threshold=args.threshold,
        output_file=args.outputFile
    )