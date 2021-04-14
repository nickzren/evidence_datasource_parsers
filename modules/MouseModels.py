#!/usr/bin/env python3
"""Evidence parser for the animal model sources from PhenoDigm."""

import argparse
import gzip
import json
import logging
import os
import pathlib
import tempfile
import urllib.request

import pyspark
import pyspark.sql.functions as pf
import requests
from retry import retry

HGNC_DATASET_URI = 'http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt'
HGNC_DATASET_FILENAME = os.path.split(HGNC_DATASET_URI)[-1]

MGI_DATASET_URI = 'http://www.informatics.jax.org/downloads/reports/MGI_Gene_Model_Coord.rpt'
MGI_DATASET_FILENAME = os.path.split(MGI_DATASET_URI)[-1]

IMPC_SOLR_HOST = 'http://www.ebi.ac.uk/mi/impc/solr/phenodigm/select'
IMPC_SOLR_TABLES = ('gene', 'gene_gene', 'mouse_model', 'disease_model_summary', 'disease_gene_summary', 'disease',
                    'ontology_ontology', 'ontology')
IMPC_FILENAME = 'impc_solr_{data_type}.json'
IMPC_SOLR_BATCH_SIZE = 100000
IMPC_SOLR_TIMEOUT = 600


class ImpcSolrRetriever:
    """Retrieves data from the IMPC SOLR API and saves the JSONs to the specified location."""

    def __init__(self, solr_host: str, timeout: int, rows: int):
        """Initialise the query parameters: SOLR endpoint to make the requests against; timeout to apply to the
        requests, in seconds; and the number of SOLR documents requested in a single batch."""
        self.solr_host = solr_host
        self.timeout = timeout
        self.rows = rows

    # The decorator ensures that the requests are retried in case of network or server errors.
    @retry(tries=3, delay=5, backoff=1.2, jitter=(1, 3))
    def query_solr(self, data_type, start):
        """Returns one batch of SOLR documents of the specified data type."""
        params = {'q': '*:*', 'fq': f'type:{data_type}', 'start': start, 'rows': self.rows}
        response = requests.get(self.solr_host, params=params, timeout=self.timeout)
        response.raise_for_status()  # Check for HTTP errors. This will be caught by @retry.
        return response.json()

    def fetch_data(self, data_type, output_filename):
        """Fetch all rows of the requested type to the specified location."""
        with open(output_filename, 'wt') as outfile:
            start, total = 0, 0  # Initialise the counters.
            while True:
                solr_data = self.query_solr(data_type, start)
                assert solr_data['response']['numFound'] != 0, 'A table requested from SOLR should not be empty.'
                for doc in solr_data['response']['docs']:  # Write data to file.
                    json.dump(doc, outfile)
                    outfile.write('\n')
                # Increment the counters.
                start += self.rows
                total += len(solr_data['response']['docs'])
                # Exit when all documents have been retrieved.
                if total == solr_data['response']['numFound']:
                    break


class PhenoDigm:
    """Retrieve and load the data, process, and then write the resulting evidence strings."""

    def __init__(self, logger):
        super(PhenoDigm, self).__init__()
        self.logger = logger
        self.spark = pyspark.sql.SparkSession.builder.appName('phenodigm_parser').getOrCreate()
        self.hgnc_gene_id_to_ensembl_human_gene_id, self.mgi_gene_id_to_ensembl_mouse_gene_id = None, None
        (self.mouse_gene_to_human_gene, self.disease_model_summary, self.mouse_model,
         self.mouse_phenotype_to_human_phenotype, self.disease) = [None] * 5
        self.evidence = None

    def update_cache(self, cache_dir):
        """Fetch the Ensembl gene ID and SOLR data into the local cache directory."""
        pathlib.Path(cache_dir).mkdir(parents=False, exist_ok=True)

        self.logger.info('Fetching human gene ID mappings from HGNC.')
        urllib.request.urlretrieve(HGNC_DATASET_URI, os.path.join(cache_dir, HGNC_DATASET_FILENAME))

        self.logger.info('Fetching mouse gene ID mappings from MGI.')
        urllib.request.urlretrieve(MGI_DATASET_URI, os.path.join(cache_dir, MGI_DATASET_FILENAME))

        self.logger.info('Fetching Phenodigm data from IMPC SOLR.')
        impc_solr_retriever = ImpcSolrRetriever(solr_host=IMPC_SOLR_HOST, timeout=IMPC_SOLR_TIMEOUT,
                                                rows=IMPC_SOLR_BATCH_SIZE)
        for data_type in IMPC_SOLR_TABLES:
            self.logger.info(f'Fetching Phenodigm data type {data_type}.')
            filename = os.path.join(cache_dir, IMPC_FILENAME.format(data_type=data_type))
            impc_solr_retriever.fetch_data(data_type, filename)

    def load_data_from_cache(self, cache_dir):
        """Load the Ensembl gene ID and SOLR data from the downloaded TSV/JSON files into Spark."""
        self.hgnc_gene_id_to_ensembl_human_gene_id = (
            self.spark.read.csv(os.path.join(cache_dir, HGNC_DATASET_FILENAME), sep='\t', header=True)
            .select('hgnc_id', 'ensembl_gene_id')
            .withColumnRenamed('hgnc_id', 'hgnc_gene_id')
            .withColumnRenamed('ensembl_gene_id', 'ensembl_human_gene_id')
        )
        self.mgi_gene_id_to_ensembl_mouse_gene_id = (
            self.spark.read.csv(os.path.join(cache_dir, MGI_DATASET_FILENAME), sep='\t', header=True)
            .withColumnRenamed('1. MGI accession id', 'mgi_gene_id')
            .withColumnRenamed('11. Ensembl gene id', 'ensembl_mouse_gene_id')
            .select('mgi_gene_id', 'ensembl_mouse_gene_id')
        )
        self.mouse_gene_to_human_gene = (
            self.spark.read.json(os.path.join(cache_dir, IMPC_FILENAME.format(data_type='gene_gene')))
            .select('gene_id', 'hgnc_gene_id')
            .withColumnRenamed('gene_id', 'mgi_gene_id')
        )
        self.disease_model_summary = (
            self.spark.read.json(os.path.join(cache_dir, IMPC_FILENAME.format(data_type='disease_model_summary')))
            .select('model_id', 'model_genetic_background', 'model_description',
                    'disease_id', 'disease_term', 'disease_model_max_norm',
                    'marker_id')
            .withColumnRenamed('marker_id', 'mgi_gene_id')
            .limit(100)
        )
        self.mouse_model = (
            self.spark.read.json(os.path.join(cache_dir, IMPC_FILENAME.format(data_type='mouse_model')))
            .select('model_id', 'model_phenotypes')
        )
        self.mouse_phenotype_to_human_phenotype = (
            self.spark.read.json(os.path.join(cache_dir, IMPC_FILENAME.format(data_type='ontology_ontology')))
            .select('mp_id', 'mp_term', 'hp_id', 'hp_term')
        )
        self.disease = (
            self.spark.read.json(os.path.join(cache_dir, IMPC_FILENAME.format(data_type='disease')))
            .select('disease_id', 'disease_phenotypes')
        )

    def generate_phenodigm_evidence_strings(self):
        # Prepare the gene mapping tables for mouse and human. Each mapping is not guaranteed to be one-to-one, so
        # appropriate aggregations are applied, and the corresponding explosions will be applied after joining the
        # table. For example (a hypothetical scenario), if a `marker_id` (MGI gene identifier) maps to 2 different
        # ENSMUSG accessions, and at the same time to 3 different ENSG accessions, a single source row will eventually
        # be exploded into 6 to reflect the total evidence available.
        mgi_to_mouse_ensembl_agg = (  # Example: MGI:1 → [ENSMUSG1, ENSMUSG2]
            self.mgi_gene_id_to_ensembl_mouse_gene_id
            .groupby('mgi_gene_id')
            .agg(pf.collect_list('ensembl_mouse_gene_id').alias('ensembl_mouse_gene_id_list'))
        )

        # For human genes, we map: MGI ID → HGNC ID → Ensembl human gene ID.
        mgi_to_hgnc = (
            self.mouse_gene_to_human_gene
            .groupby('hgnc_gene_id')
            .agg(pf.collect_list('mgi_gene_id').alias('mgi_gene_id_list'))
        )
        hgnc_to_ensg = (
            self.hgnc_gene_id_to_ensembl_human_gene_id
            .groupby('hgnc_gene_id')
            .agg(pf.collect_list('ensembl_human_gene_id').alias('ensembl_human_gene_id_list'))
        )
        mgi_to_human_ensembl_agg = (  # Example: MGI:1 → [ENSG1, ENSG2]
            mgi_to_hgnc.join(hgnc_to_ensg, on='hgnc_gene_id', how='inner')
            .drop('hgnc_gene_id')  # Thank you for making the join possible, but we don't need you anymore.
            .withColumn('mgi_gene_id', pf.explode(pf.column('mgi_gene_id_list')))
            .withColumn('ensembl_human_gene_id', pf.explode(pf.column('ensembl_human_gene_id_list')))
            .drop('mgi_gene_id_list', 'ensembl_human_gene_id_list')
            .groupby('mgi_gene_id').agg(pf.collect_list('ensembl_human_gene_id').alias('ensembl_human_gene_id_list'))
        )

        # Process phenotype information
        model_phenotypes = (  # model ID, MP ID, MP term, HP ID, HP term
            self.mouse_model
            .withColumn('phenotype', pf.explode('model_phenotypes'))
            .withColumn('mp_id', pf.split(pf.col('phenotype'), ' ').getItem(0))
            .drop('model_phenotypes')
            .join(self.mouse_phenotype_to_human_phenotype, on='mp_id', how='left')
        )
        diseases = (  # disease ID, HP ID
            self.disease
            .withColumn('phenotype', pf.explode('disease_phenotypes'))
            .withColumn('hp_id', pf.split(pf.col('phenotype'), ' ').getItem(0))
        )
        # For human phenotypes, we only want to include the ones which are present in the disease *and* also can be
        # traced back to the model phenotypes through the MP → HP mapping relationship.
        matched_human_phenotypes = (
            self.disease_model_summary
            .join(model_phenotypes, on='model_id', how='inner')
            .join(diseases, on=['disease_id', 'hp_id'], how='inner')
            .select('model_id', 'disease_id', 'hp_id', 'hp_term')
            .groupby('model_id', 'disease_id')
            .agg(
                pf.collect_set(pf.struct(
                    pf.col('hp_id').alias('id'),
                    pf.col('hp_term').alias('label')
                )).alias('diseaseModelAssociatedHumanPhenotypes')
            )
            .select('model_id', 'disease_id', 'diseaseModelAssociatedHumanPhenotypes')
        )
        all_mouse_phenotypes = (
            model_phenotypes
            .select('model_id', 'mp_id', 'mp_term')
            .groupby('model_id')
            .agg(
                pf.collect_set(pf.struct(
                    pf.col('mp_id').alias('id'),
                    pf.col('mp_term').alias('label')
                )).alias('diseaseModelAssociatedModelPhenotypes')
            )
            .select('model_id', 'diseaseModelAssociatedModelPhenotypes')
        )

        self.evidence = (
            self.disease_model_summary

            # Add the gene mapping information. Note the mappings are not one-to-one in general.
            .join(mgi_to_mouse_ensembl_agg, on='mgi_gene_id', how='inner')
            .join(mgi_to_human_ensembl_agg, on='mgi_gene_id', how='inner')
            .withColumn('ensembl_mouse_gene_id', pf.explode(pf.column('ensembl_mouse_gene_id_list')))
            .withColumn('ensembl_human_gene_id', pf.explode(pf.column('ensembl_human_gene_id_list')))
            .drop('ensembl_mouse_gene_id_list', 'ensembl_human_gene_id_list')

            # Add phenotype information
            .join(matched_human_phenotypes, on=['model_id', 'disease_id'], how='left')
            .join(all_mouse_phenotypes, on='model_id', how='left')

            # Rename some columns
            .withColumnRenamed('disease_id', 'diseaseFromSourceId')
            .withColumnRenamed('disease_term', 'diseaseFromSource')
            .withColumnRenamed('ensembl_human_gene_id', 'targetFromSourceId')
            .withColumnRenamed('ensembl_mouse_gene_id', 'targetInModel')
            .withColumnRenamed('model_description', 'biologicalModelAllelicComposition')
            .withColumnRenamed('model_genetic_background', 'biologicalModelGeneticBackground')

            # Strip trailing modifiers from the model ID
            # For example: MGI:6274930#hom#early → MGI:6274930
            .withColumn(
                'biologicalModelId',
                pf.split(pf.col('model_id'), '#').getItem(0)
            )

            # Convert the percentage score into fraction
            .withColumn('resourceScore', pf.col('disease_model_max_norm') / 100.0)

            # Remove intermediate columns
            .drop('disease_model_max_norm', 'model_id')

            # Add constant value columns
            .withColumn('datasourceId', pf.lit('phenodigm'))
            .withColumn('datatypeId', pf.lit('animal_model'))
        )

    def write_evidence_strings(self, filename):
        """Dump the Spark evidence dataframe into a temporary directory as separate JSON chunks. Collect and combine the
        chunks into the final output file. JSON keys are sorted in the process, but the order of the output records
        themselves is not modified."""
        with tempfile.TemporaryDirectory() as tmp_dir_name, gzip.open(filename, 'wt') as outfile:
            self.evidence.write.format('json').mode('overwrite').save(tmp_dir_name)
            json_chunks = [f for f in os.listdir(tmp_dir_name) if f.endswith('.json')]
            for chunk_filename in json_chunks:
                with open(os.path.join(tmp_dir_name, chunk_filename), 'rt') as json_chunk:
                    for line in json_chunk:
                        outfile.write(json.dumps(json.loads(line), sort_keys=True) + '\n')

    def process_all(self, cache_dir, output, use_cached):
        if not use_cached:
            self.logger.info('Update the HGNC/MGI/SOLR cache')
            self.update_cache(cache_dir)

        self.logger.info('Load gene mappings and SOLR data from local cache')
        self.load_data_from_cache(cache_dir)

        self.logger.info('Build the evidence strings.')
        self.generate_phenodigm_evidence_strings()

        self.logger.info('Collect and write the evidence strings.')
        self.write_evidence_strings(output)


def main(cache_dir, output, use_cached=False, log_file=None):
    # Initialize the logger based on the provided log file. If no log file is specified, logs are written to STDERR.
    logging_config = {
        'level': logging.INFO,
        'format': '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }
    if log_file:
        logging_config['filename'] = log_file
    logging.basicConfig(**logging_config)

    # Process the data.
    PhenoDigm(logging).process_all(cache_dir, output, use_cached)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', help='Directory to store the HGNC/MGI/SOLR cache files in.', required=True)
    parser.add_argument('--output', help='Name of the json.gz file to output the evidence strings into.', required=True)
    parser.add_argument('--use-cached', help='Use the existing cache and do not update it.', action='store_true')
    parser.add_argument('--log-file', help='Optional filename to redirect the logs into.')
    args = parser.parse_args()
    main(args.cache_dir, args.output, args.use_cached, args.log_file)
