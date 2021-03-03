# General settings that all parsers can share

import os
import pkg_resources as res
from datetime import datetime

def file_or_resource(fname=None):
    # get filename and check if in getcwd then get from the package resources folder
    filename = os.path.expanduser(fname)

    resource_package = __name__
    resource_path = '/'.join(('resources', filename))

    if filename is not None:
        abs_filename = os.path.join(os.path.abspath(os.getcwd()), filename) \
                       if not os.path.isabs(filename) else filename

        return abs_filename if os.path.isfile(abs_filename) \
            else res.resource_filename(resource_package, resource_path)

class Config:
    # shared settings

    # schema version
    VALIDATED_AGAINST_SCHEMA_VERSION = '1.2.8'

    GOOGLE_DEFAULT_PROJECT = 'open-targets'
    GOOGLE_BUCKET_EVIDENCE_INPUT = 'otar000-evidence_input'

    # UKBIOBANK
    UKBIOBANK_FILENAME = file_or_resource('ukbiobank.txt')
    UKBIOBANK_EVIDENCE_FILENAME = 'ukbiobank-30-04-2018.json'

    # IntoGEN
    INTOGEN_DRIVER_GENES_FILENAME = file_or_resource('intogen_Compendium_Cancer_Genes.tsv')
    INTOGEN_EVIDENCE_FILENAME = 'intogen-02-02-2020.json'
    INTOGEN_CANCER2EFO_MAPPING_FILENAME = file_or_resource('intogen_cancer2EFO_mapping.tsv')
    INTOGEN_COHORTS = file_or_resource('intogen_cohorts.tsv')

    # mapping that we maintain in Zooma
    OMIM_TO_EFO_MAP_URL = 'https://raw.githubusercontent.com/opentargets/platform_semantic/master/resources/xref_mappings/omim_to_efo.txt'
    ZOOMA_TO_EFO_MAP_URL = 'https://raw.githubusercontent.com/opentargets/platform_semantic/master/resources/zooma/cttv_indications_3.txt'

    # mouse models (Phenodigm)
    # used to be 'http://localhost:8983' # 'solrclouddev.sanger.ac.uk'
    MOUSEMODELS_PHENODIGM_SOLR = 'http://www.ebi.ac.uk/mi/impc'
    # write to the cloud direcly
    MOUSEMODELS_CACHE_DIRECTORY = 'PhenoDigm/phenodigmcache'
    MOUSEMODELS_EVIDENCE_FILENAME = f"phenodigm-{datetime.today().strftime('%Y-%m-%d')}.json.gz"
