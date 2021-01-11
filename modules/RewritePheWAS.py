from common.HGNCParser import GeneParser
import logging
import requests
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
import argparse

class phewasEvidenceGenerator():
    def __init__(self, inputFile, mappingStep, schemaVersion):
        # Create spark session     
        self.spark = SparkSession.builder \
                .appName('evidence_builder') \
                .getOrCreate()

        # Build JSON schema url from version # TODO: Remove reference to schema
        self.schemaVersion = schemaVersion
        schema_url = "https://raw.githubusercontent.com/opentargets/json_schema/ds_1249_new_json_schema/draft4_schemas/opentargets.json" #TODO Update the url 
        logging.info(f"Loading JSON Schema from {schema_url}")

        # Initialize mapping variables
        self.mappingStep = mappingStep
        self.mappingsFile = "https://raw.githubusercontent.com/opentargets/mappings/master/phewascat.mappings.tsv"

        # Data extracted from Varient Index
        self.consequencesFile = "https://storage.googleapis.com/otar000-evidence_input/PheWAS/data_files/phewas_w_consequences.csv"

        # Adding remote files to Spark Context
        self.spark.sparkContext.addFile(self.mappingsFile)
        self.spark.sparkContext.addFile(self.consequencesFile)
    
        # Initialize input files
        self.inputFile = inputFile
        
        # Initialize gene parser
        gene_parser = GeneParser()
        gene_parser._get_hgnc_data_from_json()
        self.udfGeneParser = udf(
            lambda X: gene_parser.genes[X.strip("*")],
            StringType()
        )

        self.dataframe = None
        self.enrichedDataframe = None
        self.one2manyVariants = None
        
    def writeEvidenceFromSource(self):
        '''
        Processing of the dataframe to build all the evidences from its data
        Returns:
            evidences (array): Object with all the generated evidences strings from source file
        '''
        # Read input file
        self.dataframe = self.spark.read.csv(self.inputFile, header=True)

        # Filter out null genes & p-value > 0.05
        self.dataframe = self.dataframe \
                        .filter(col("gene").isNotNull()) \
                        .filter(col("p") < 0.05)

        # Mapping step
        if self.mappingStep:
            phewasMapping = self.spark.read.csv(SparkFiles.get("phewascat.mappings.tsv"), sep=r'\t', header=True)
            self.dataframe = self.dataframe.join(
                phewasMapping,
                on=["Phewas_string"],
                how="inner"
            )
        
        self.dataframe = self.dataframe.withColumn(
            "gene",
            self.udfGeneParser(col("gene"))
        )

        # Get functional consequence per variant from OT Genetics Portal
        self.enrichedDataframe = self.enrichVariantData()

        # Build evidence strings per row
        evidences = self.dataframe.rdd \
            .map(phewasEvidenceGenerator.parseEvidenceString) \
            .collect() # list of dictionaries

    def enrichVariantData(self):
        phewasWithConsequences = self.spark.read.csv(SparkFiles.get("phewas_w_consequences.csv"), header=True)
        phewasWithConsequences = phewasWithConsequences \
                                        .withColumnRenamed("rsid", "snp") \
                                        .withColumnRenamed("gene_id", "gene")

        # Enriching dataframe with consequences --> more records due to 1:many associations
        self.dataframe = self.dataframe.join(
            phewasWithConsequences,
            on=["gene", "snp"],
            how="inner"
        )

        # Building variantId: "chrom_pos_ref_alt" of the respective rsId
        # If one rsId has several variants, variantId = none
        one2manyVariants_df = phewasWithConsequences \
                                    .groupBy("snp") \
                                    .agg(count("snp")) \
                                    .filter(col("count(snp)") > 1)
        self.one2manyVariants = list(one2manyVariants_df.toPandas()["snp"])
        self.enrichedDataframe = self.dataframe.rdd.map(self.writeVariantId).toDF()
        
        return self.enrichedDataframe

    def writeVariantId(row):
        rd = row.asDict()
        if row["snp"] not in one2manyVariants:
            rd["variantId"] = "{}_{}_{}_{}".format(row["chrom"], row["pos"], row["ref"], row["alt"])
        else:
            rd["variantId"] = None
        new_row = Row(**rd)
        return new_row

    @staticmethod
    def parseEvidenceString(row):
        try:
            evidence = {
                datasourceId : "phewas_catalog",
                datatypeId : "genetic_association",
                diseaseFromSource : row["phewas_string"],
                diseaseFromSourceId : row["phewas_code"],
                diseaseFromSourceMappedId : row["EFO_id"].split("/")[-1],
                oddsRatio : row["odds_ratio"],
                resourceScore : row["p"],
                studyCases : row["cases"],
                targetFromSourceId : row["gene"].strip("*"),
                variantFunctionalConsequenceId : row["consequence_link"].split("/")[-1] if row["consequence_link"] else "SO_0001060",
                variantId : row["variantId"],
                variantRsId : row["snp"]
            }
            return evidence
        except Exception as e:
            raise        

def main():
    # Initiating parser
    parser = argparse.ArgumentParser(description=
    "This script generates evidences from the PheWAS Catalog data source.")

    parser.add_argument("-i", "--inputFile", required=True, type=str, help="Input .csv file with the table containing association details.")
    parser.add_argument("-o", "--outputFile", required=True, type=str, help="Name of the json output file containing the evidence strings.")
    parser.add_argument("-s", "--schemaVersion", required=True, type=str, help="JSON schema version to use, e.g. 1.6.9. It must be branch or a tag available in https://github.com/opentargets/json_schema.")
    parser.add_argument("-m", "--mappingStep", required=False, type=bool, default=True, help="State whether to run the disease to EFO term mapping step or not.")

    # Parsing parameters
    args = parser.parse_args()

    dataframe = args.inputFile
    outputFile = args.outputFile
    schemaVersion = args.schemaVersion
    mappingStep = args.mappingStep

    # Initialize logging:
    logging.basicConfig(
    filename='evidence_builder.log',
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Initialize evidence builder object
    evidenceBuilder = phewasEvidenceGenerator(dataframe, mappingStep, schemaVersion)

    # Writing evidence strings into a json file
    evidences = evidenceBuilder.writeEvidenceFromSource()

    with open(outputFile, "wt") as f:
        for evidence in evidences:
            json.dump(evidence, f)
            f.write('\n')
    logging.info(f"Evidence strings saved into {outputFile}. Exiting.")

if __name__ == '__main__':
    main()
    
