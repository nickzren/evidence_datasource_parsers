#!/usr/bin/env python3
"""Parser for data submitted from the Validation Lab."""

import argparse
import json
import logging
import sys
from functools import reduce

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, lit, struct, udf, when, collect_list
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.dataframe import DataFrame

from common.evidence import detect_spark_memory_limit, write_evidence_strings

# This is a map that provides recipie to generate the biomarker objects
# If a value cannot be found in the map, the value will be returned.
BIOMARKERMAPS = {
    'MS_status': {
        'direct_mapping': {
            "MSI": {
                "name": "MSI",
                "description": "Microsatellite instable"
            },
            "MSS": {
                "name": "MSS",
                "description": "Microsatellite stable"
            }
        }
    },
    'CRIS_subtype': {
        "direct_mapping": {
            "A": {
                "name": "CRIS-A",
                "description": "mucinous, glycolytic, enriched for microsatellite instability or KRAS mutations"
            },
            "B": {
                "name": "CRIS-B",
                "description": "TGF-β pathway activity, epithelial-mesenchymal transition, poor prognosis"
            },
            "C": {
                "name": "CRIS-C",
                "description": "elevated EGFR signalling, sensitivity to EGFR inhibitors"
            },
            "D": {
                "name": "CRIS-D",
                "description": "WNT activation, IGF2 gene overexpression and amplification"
            },
            "E": {
                "name": "CRIS-E",
                "description": "Paneth cell-like phenotype, TP53 mutations."
            }
        }
    },
    'KRAS_status': {
        'description': 'KRAS mutation status: ',
        'name': 'KRAS-',
    },
    'TP53_status': {
        'description': 'TP53 mutation status: ',
        'name': 'TP53-',
    },
    'APC_status': {
        'description': 'APC mutation status: ',
        'name': 'APC-',
    }
}


@ udf(StructType([
    StructField("hypothesis", StringType(), False),
    StructField("description", StringType(), False)
]))
def parse_hypothesis(biomarker: str, biomarker_status: str, hypothesis: str) -> dict:
    """This function parses the fields provided by the validation lab and returns with the hypothesis object."""

    # This is a spacer at the moment, and we just return the biomarker and the description:
    return {
        'hypothesis': f'{biomarker}-{biomarker_status}',
        'description': hypothesis
    }

@ udf(StructType([
    StructField("name", StringType(), False),
    StructField("description", StringType(), False)
]))
def get_biomarker(columnName, biomarker):
    '''This function returns with a struct with the biomarker name and description'''

    # Question marks signs missing biomarker status:
    if biomarker == '?':
        return None

    # If the biomarker has a direct mapping:
    if 'direct_mapping' in BIOMARKERMAPS[columnName]:
        try:
            return BIOMARKERMAPS[columnName]['direct_mapping'][biomarker]
        except KeyError:
            logging.warning(
                f'Could not find direct mapping for {columnName}:{biomarker}')
            return None

    # If the value needs to be parsed:
    if biomarker == 'wt':
        return {
            'name': BIOMARKERMAPS[columnName]['name'] + biomarker,
            'description': BIOMARKERMAPS[columnName]['description'] + 'wild type'
        }
    elif biomarker == 'mut':
        return {
            'name': BIOMARKERMAPS[columnName]['name'] + biomarker,
            'description': BIOMARKERMAPS[columnName]['description'] + 'mutant'
        }
    else:
        logging.warning(
            f'Could not find direct mapping for {columnName}:{biomarker}')
        return None


def parse_experimental_parameters(parmeter_file: str) -> dict:
    """
    Parse experimental parameters from a file.

    Args:
        parmeter_file: Path to a file containing experimental parameters.

    Returns:
        A dictionary of experimental parameters.
    """
    with open(parmeter_file, 'r') as f:
        return json.load(f)


def initialize_sparksession() -> SparkSession:
    """Initialize spark session."""
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
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    return spark


def get_cell_passport_data(spark: SparkSession, cell_passport_file: str) -> DataFrame:

    # loading cell line annotation data from Sanger:
    return (
        spark.read
        .option("multiline", True)
        .csv(cell_passport_file, header=True, sep=',', quote='"')
        .select(
            col('model_name').alias('name'),
            col('model_id').alias('id'),
            col('tissue')
        )
        # Some model names needs to be changed to match the Validation lab dataset:
        .withColumn(
            'name',
            when(col('name') == 'HT-29', 'HT29')
            .when(col('name') == 'HCT-116', 'HCT116')
            .when(col('name') == 'LS-180', 'LS180')
            .otherwise(col('name'))
        )
        .persist()
    )


def parse_experiment(spark: SparkSession, parameters: dict, cellPassportDf: DataFrame) -> DataFrame:
    """
    Parse experiment data from a file.

    Args:
        spark: Spark session.
        parameters: Dictionary of experimental parameters.
        cellPassportDf: Dataframe of cell passport data.

    Returns:
        A dataframe of experiment data.
    """

    # Extracting parameters:
    experimentFile = parameters['experimentData']
    contrast = parameters['contrast']
    studyOverview = parameters['studyOverview']
    projectId = parameters['projectId']
    projectDescription = parameters['projectDescription']
    diseaseFromSource = parameters['diseaseFromSource']
    diseaseFromSourceMapId = parameters['diseaseFromSourceMappedId']
    confidenceCutoff = parameters['confidenceCutoff']
    cellLineFile = parameters['cellLineFile']
    hypothesisFile = parameters['hypothesisFile']

    # Cell line data:
    # Reading cell metadata from validation lab:
    validation_lab_cell_lines = (
        spark.read.csv(cellLineFile, sep='\t', header=True)

        # Renaming columns:
        .withColumnRenamed('CO_line', 'name')

        # Joining dataset with cell model data read downloaded from Sanger website:
        .join(cellPassportDf, on='name', how='left')

        # Adding UBERON code to tissues (it's constant colon)
        .withColumn('tissueID', lit('UBERON_0000059'))

        # generating disease cell lines object:
        .withColumn(
            'diseaseCellLines',
            array(struct(col('name'), col('id'), col('tissue'), col('tissueId')))
        )
        .drop(*['id', 'tissue', 'tissueId'])
        .persist()
    )

    logging.info(
        f'Validation lab cell lines has {validation_lab_cell_lines.count()} cell types.')

    # Defining how to process biomarkers:
    # 1. Looping through all possible biomarker - from biomarkerMaps.keys()
    # 2. The biomakers are then looked up in the map and process based on how the map defines.
    # 3. Description is also added read from the map.
    expressions = map(
        # Function to process biomarker:
        lambda biomarker: (biomarker, get_biomarker(
            lit(biomarker), col(biomarker))),

        # Iterator to apply the function over:
        BIOMARKERMAPS.keys()
    )

    # Applying the full map on the dataframe one-by-one:
    biomarkers = reduce(lambda DF, value: DF.withColumn(
        *value), expressions, validation_lab_cell_lines)

    # Pooling together all the biomarker structures into one single array:
    biomarkers = (
        biomarkers
        .select('name', array(*BIOMARKERMAPS.keys()).alias('biomarkers'))
    )

    # Joining biomarkers with cell line data:
    validation_lab_cell_lines = (
        validation_lab_cell_lines
        .join(biomarkers, on='name', how='inner')

        # Dropping biomarker columns:
        .drop(*list(BIOMARKERMAPS.keys()))
        .persist()
    )

    # Reading and processing hypothesis data:
    hypothesis = (
        spark.read.csv(hypothesisFile, sep='\t', header=True)
        .withColumn('hypothesis', parse_hypothesis(col('biomarker'), col('biomarkerStatus'), col('description')))
        .groupBy('gene')
        .agg(collect_list('hypothesis').alias('validationHypotheses'))
        .persist()
    )

    print(hypothesis.printSchema())

    # Reading experiment data from validation lab:
    evidence = (
        # Reading evidence:
        spark.read.csv(experimentFile, sep='\t', header=True)

        # Joining hypothesis data:
        .join(hypothesis, on='gene', how='left')

        # Rename existing columns need to be updated:
        .withColumnRenamed('gene', 'targetFromSource')
        .withColumnRenamed('cell-line', 'name')

        # Parsing resource score:
        .withColumn('resourceScore', col('effect-size').cast("double"))

        # Generate the binary confidence calls:
        .withColumn(
            'confidence',
            when(col('resourceScore') >= confidenceCutoff, lit('significant'))
            .otherwise(lit('not significant'))
        )
        .withColumn(
            'expectedConfidence',
            when(col('expected-to-pass') == 'TRUE', lit('significant'))
            .otherwise(lit('not significant'))
        )

        # Adding constants:
        .withColumn('statisticalTestTail', lit('upper tail'))
        .withColumn('contrast', lit(contrast))
        .withColumn('studyOverview', lit(studyOverview))

        # This column is specific for this dataset:
        .withColumn('datasourceId', lit('ot_crispr_validation'))
        .withColumn('datatypeId', lit('ot_validation_lab'))
        .withColumn("diseaseFromSourceMappedId", lit(diseaseFromSourceMapId))
        .withColumn("diseaseFromSource", lit(diseaseFromSource))

        # This should be added to the crispr dataset as well:
        .withColumn('projectId', lit(projectId))
        .withColumn('projectDescription', lit(projectDescription))

        # Joining cell line data:
        .join(validation_lab_cell_lines, on='name', how='left')

        # Drop unused columns:
        .drop(*['name', 'pass-fail', 'expected-to-pass', 'effect-size'])
    )

    logging.info(f'Evidence count: {evidence.count()}.')
    return evidence


def main(inputFile: str, outputFile: str) -> None:

    # Initialize spark session
    spark = initialize_sparksession()

    # Parse experimental parameters:
    parameters = parse_experimental_parameters(inputFile)

    # Opening and parsing the cell passport data from Sanger:
    cell_passport_df = get_cell_passport_data(
        spark, parameters['sharedParemeters']['cellPassportFile'])

    logging.info(
        f'Cell passport dataframe has {cell_passport_df.count()} rows.')

    logging.info('Parsing experiment data...')
    for experiment in parameters['experiments']:
        evidence_df = parse_experiment(spark, experiment, cell_passport_df)
        write_evidence_strings(evidence_df, outputFile)


if __name__ == '__main__':

    # Reading output file name from the command line:
    parser = argparse.ArgumentParser(
        description='This script fetches TEP data from Structural Genomics Consortium.')
    parser.add_argument('--output_file', '-o', type=str,
                        help='Output file. gzipped JSON', required=True)
    parser.add_argument('--input_file', '-i', type=str,
                        help='A JSON file describing exeriment metadata', required=True)
    parser.add_argument('--log_file', type=str,
                        help='File into which the logs are saved', required=False)
    args = parser.parse_args()

    # If no logfile is specified, logs are written to the standard error:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if args.log_file:
        logging.config.fileConfig(filename=args.log_file)
    else:
        logging.StreamHandler(sys.stderr)

    # Passing all the required arguments:
    main(args.input_file, args.output_file)
