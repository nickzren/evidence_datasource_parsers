import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from common.evidence import (
    GenerateDiseaseCellLines,
    initialize_logger,
    initialize_sparksession,
    write_evidence_strings,
)

# This parser is specific for the second release of Porject Score, so the publication identifier is hardcoded
PMID = "38215750"


def generate_project_score_evidence(
    project_score_cell_lines: DataFrame, project_score_evidence: DataFrame
) -> DataFrame:
    """Generate evidence strings for Project Score v2.

    Args:
        project_score_cell_lines (DataFrame): DataFrame containing cell line annotations
        project_score_evidence (DataFrame): DataFrame containing evidence data

    Returns:
        DataFrame: DataFrame containing evidence strings
    """
    pmid = "38215750"
    return (
        project_score_evidence.select(
            f.col("targetSymbol").alias("targetFromSource"),
            f.col("diseaseName").alias("diseaseFromSource"),
            f.col("diseaseId").alias("diseaseFromSourceMappedId"),
            f.col("PRIORITY").cast(t.FloatType()).alias("resourceScore"),
            f.col("targetId").alias("targetFromSourceId"),
            f.array(f.lit(pmid)).alias("literature"),
            f.lit("crispr").alias("datasourceId"),
            f.lit("affected_pathway").alias("datatypeId"),
            f.lower(f.col("cancerType")).alias("cancerType"),
        )
        .join(project_score_cell_lines, on="cancerType", how="left")
        # Dropping the pancancer genes:
        .filter(f.col("cancerType") != "pancancer")
        # Cleaning table:
        .drop("cancerType")
    )


def main(
    spark: SparkSession,
    evid_file: str,
    cell_line_file: str,
    cell_passport_file: str,
    cell_line_to_uberon_mapping: str,
    out_file: str,
):
    # Extract disease cell line data from cell passport file:
    cell_passport_data = GenerateDiseaseCellLines(
        spark, cell_passport_file, cell_line_to_uberon_mapping
    )

    passport_disease_cell_lines = cell_passport_data.generate_disease_cell_lines()

    # Joining disease cell-lines dataframe for Project Score:
    disease_cell_lines = (
        spark.read.csv(cell_line_file, sep="\t", header=True)
        .select(
            f.lower(f.col("CANCER_TYPE")).alias("cancerType"),
            f.col("CMP_ID").alias("id"),
        )
        .join(passport_disease_cell_lines, on="id", how="left")
        .groupBy("cancerType")
        .agg(f.collect_set("diseaseCellLine").alias("diseaseCellLines"))
    )

    # Read gene based data and generate evidence strings:
    evidence_table = spark.read.csv(evid_file, sep="\t", header=True)

    project_score_evidence = generate_project_score_evidence(
        disease_cell_lines, evidence_table
    )

    write_evidence_strings(project_score_evidence, out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse datasource parsers from Project Score"
    )
    parser.add_argument(
        "--descriptions_file",
        help="Name of tsv file with the description of the method per cancer type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--evidence_file",
        help="Name of tsv file with the priority score",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cell_types_file",
        help="Name of tsv file with cell line names per cancer type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="Name of evidence file. (gzip compressed json)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log_file",
        help="Name of log file. If not provided logs are written to standard error.",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments:
    args = parse_args()
    desc_file = args.descriptions_file
    evid_file = args.evidence_file
    cell_file = args.cell_types_file
    out_file = args.output_file

    # Initialize logger:
    initialize_logger()

    # Initialize spark session:
    spark = initialize_sparksession()

    main(desc_file, evid_file, cell_file, out_file)
    initialize_logger()

    main(desc_file, evid_file, cell_file, out_file)
    main(desc_file, evid_file, cell_file, out_file)
    initialize_logger()

    main(desc_file, evid_file, cell_file, out_file)
    main(desc_file, evid_file, cell_file, out_file)
