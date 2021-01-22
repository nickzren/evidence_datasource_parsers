import argparse
import gzip
import logging
import json
from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class progenyEvidenceGenerator():
    def __init__(self):
        # Create spark session     
        self.spark = (SparkSession.builder
                .appName('progeny')
                .getOrCreate())
    
        # Initialize source table
        self.dataframe = None

    def generateEvidenceFromSource(self, inputFile, skipMapping):
        '''
        Processing of the input file to build all the evidences from its data
        Returns:
            evidences (array): Object with all the generated evidences strings from source file
        '''
        # Read input file
        self.dataframe = (self.spark.read
                                .option("header", "true")
                                .option("delimiter", "\t")
                                .option("inferSchema", "true")
                                .csv(inputFile))

        # Mapping step
        if not skipMapping:
            self.dataframe = self.cancer2EFO()
        
        self.dataframe = self.pathway2Reactome()

        # Build evidence strings per row
        evidences = (self.dataframe.rdd
            .map(progenyEvidenceGenerator.parseEvidenceString)
            .collect() # list of dictionaries
            )
        
        return evidences
    
    def cancer2EFO(self):
        diseaseMappingsFile = (self.spark
                        .read.csv("resources/cancer2EFO_mappings.tsv", sep=r'\t', header=True)
                        .select("Cancer_type_acronym", "EFO_id")
                        .withColumnRenamed("Cancer_type_acronym", "Cancer_type"))

        self.dataframe = self.dataframe.join(
            diseaseMappingsFile,
            on="Cancer_type",
            how="inner"
)

        return self.dataframe
    
    def pathway2Reactome(self):
        pathwayMappingsFile = (self.spark
                        .read.csv("resources/pathway2Reactome_mappings.tsv", sep=r'\t', header=True)
                        .withColumnRenamed("pathway", "Pathway"))
        
        self.dataframe = (self.dataframe
                .join(
                    pathwayMappingsFile,
                    on="Pathway",
                    how="inner"
                )
                .withColumn(
                    "target",
                    split(col("target"), ", ") 
                )
                .withColumn(
                    "target",
                    explode("target")
                ))

        return self.dataframe

    @staticmethod
    def parseEvidenceString(row):
        try:
            evidence = {
                "datasourceId" : "progeny",
                "datatypeId" : "affected_pathway",
                "diseaseFromSource" : row["Cancer_type"],
                "resourceScore" : row["P.Value"],
                "pathwayName" : row["description"],
                "pathwayId" : row["reactomeId"],
                "targetFromSourceId" : row["target"]
            }
            if "EFO_id" in row:
                evidence["diseaseFromSourceMappedId"] = row["EFO_id"]
            return evidence
        except Exception:
            raise        

def main():
    # Initiating parser
    parser = argparse.ArgumentParser(description=
    "This script generates evidences for the PROGENy data source.")

    parser.add_argument("-i", "--inputFile", required=True, type=str, help="Input source .txt file.")
    parser.add_argument("-o", "--outputFile", required=True, type=str, help="Name of the evidence compressed JSON file containing the evidence strings.")
    parser.add_argument("-s", "--skipMapping", required=False, action="store_true", help="State whether to skip the disease to EFO mapping step.")

    # Parsing parameters
    args = parser.parse_args()

    inputFile = args.inputFile
    outputFile = args.outputFile
    skipMapping = args.skipMapping

    # Initialize logging:
    logging.basicConfig(
    filename='evidence_builder.log',
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Initialize evidence builder object
    evidenceBuilder = progenyEvidenceGenerator()

    # Writing evidence strings into a json file
    evidences = evidenceBuilder.generateEvidenceFromSource(inputFile, skipMapping)

    with gzip.open(outputFile, "wt") as f:
        for evidence in evidences:
            json.dump(evidence, f)
            f.write('\n')
    logging.info(f"{len(evidences)} evidence strings saved into {outputFile}. Exiting.")

if __name__ == '__main__':
    main()