from datetime import datetime

# Current time in YYYY-MM-DD-hh-mm format:
timeStamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Configuration is read from the config yaml::
configfile: 'configuration.yaml'

# schema version is now hardcoded, version will be read from the command line later:
schema_file = config['global']['schema_file']

##
## Copy all imput files from google buckets to the VM:
##
rule copyGeneticsStudyFile:
    input:
        config['GeneticsPortal']['study']
    output:
        '/tmp/genetics/'+config['GeneticsPortal']['study'].split('/')[-1]
    shell:
        'gsutil cp -r {input} {output}'

rule copyGeneticsVariantIndexFile:
    params:
        source=config['GeneticsPortal']['variantIndex'],
        output_folder='/tmp/genetics/'
    shell:
        'gsutil cp -r {params.source} {params.output_folder}'


rule copyGeneticsToplociFile:
    params:
        source=config['GeneticsPortal']['toploci'],
        output_folder='/tmp/genetics/'
    shell:
        'gsutil cp -r {params.source} {params.output_folder}'

rule copyGeneticsLocus2geneFile:
    params:
        source=config['GeneticsPortal']['locus2gene'],
        output_folder='/tmp/genetics/'
    shell:
        'gsutil cp -r {params.source} {params.output_folder}'

rule copyGeneticsEcoCodesFile:
    params:
        source=config['GeneticsPortal']['ecoCodes'],
        output_folder='/tmp/genetics/'
    shell:
        'gsutil cp -r {params.source} {params.output_folder}'

## Running genetics portal evidence generation. 
## * input files under opentargets-genetics project area
## * output is generated in opentargets-platform bucket.    
rule GeneticsPortal:
    params:
        locus2gene='/temp/genetics/' + config['GeneticsPortal']['locus2gene'].split('/')[-1],
        toploci='/temp/genetics/' + config['GeneticsPortal']['toploci'].split('/')[-1],
        study='/temp/genetics/' + config['GeneticsPortal']['study'].split('/')[-1],
        variantIndex='/temp/genetics/' + config['GeneticsPortal']['variantIndex'].split('/')[-1],
        ecoCodes='/temp/genetics/' + config['GeneticsPortal']['ecoCodes'].split('/')[-1],
        outputFile=config['GeneticsPortal']['outputFile'],
        threshold=config['GeneticsPortal']['threshold']
    output:
        (config['GeneticsPortal']['outputFile'] % timeStamp)
    threads: 32
    log:
        f'logs/GeneticsPortal.log'
    shell:
        '''
        python modules/GeneticsPortal_prepare_data.py \
            --locus2gene {params.locus2gene} \
            --toploci {params.toploci} \
            --study {params.study} \
            --variantIndex {params.variantIndex} \
            --ecoCodes {params.ecoCodes} \
            --outputFile {output}
        '''

# Walidating gentics portal evidence:
rule GeneticsPortal_validation:
    input:
        (config['GeneticsPortal']['outputFile'] % timeStamp)
    log:
        f'logs/GeneticsPortal.log'
    shell:
        '''
        zcat {input}/*.json.gz  | opentargets_validator --schema {schema_file}
        '''

