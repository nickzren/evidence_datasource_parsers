#!/usr/bin/env python
"""This module puts together data from different sources that describe target safety liabilities."""

import argparse
from functools import reduce
import logging
import sys
from typing import Optional

from pyspark import SparkFiles
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

from common.evidence import initialize_sparksession, write_evidence_strings


def main(
    toxcast: str,
    output: str,
    adverse_events: str,
    safety_risk: str,
    aopwiki: str,
    log_file: Optional[str] = None,
):
    """
    This module puts together data from different sources that describe target safety liabilities.

    Args:
        adverse_events: Input TSV containing adverse events associated with targets that have been collected from relevant publications. Fetched from GitHub.
        safety_risk: Input TSV containing cardiovascular safety liabilities associated with targets that have been collected from relevant publications. Fetched from GitHub.
        toxcast: Input table containing biological processes associated with relevant targets that have been observed in toxicity assays.
        output: Output gzipped json file following the target safety liabilities data model.
        log_file: Destination of the logs generated by this script. Defaults to None.
    """

    # Logger initializer. If no log_file is specified, logs are written to stderr
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    if log_file:
        logging.config.fileConfig(filename=log_file)
    else:
        logging.StreamHandler(sys.stderr)

    # Initialize spark context
    global spark
    spark = initialize_sparksession()
    spark.sparkContext.addFile(adverse_events)
    spark.sparkContext.addFile(safety_risk)
    logging.info('Remote files successfully added to the Spark Context.')

    # Load and process the input files into dataframes
    ae_df = process_adverse_events(SparkFiles.get(adverse_events.split('/')[-1]))
    sr_df = process_safety_risk(SparkFiles.get(safety_risk.split('/')[-1]))
    toxcast_df = process_toxcast(toxcast)
    aopwiki_df = process_aop(aopwiki)
    logging.info('Data has been processed. Merging...')

    # Combine dfs and group evidence
    evidence_unique_cols = [
        'id',
        'targetFromSourceId',
        'event',
        'eventId',
        'datasource',
        'effects',
        'isHumanApplicable',
        'literature',
        'url'
    ]
    safety_dfs = [ae_df, sr_df, toxcast_df, aopwiki_df]
    safety_df = (
        reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), safety_dfs)
        # Collect biosample and study metadata by grouping on the unique evidence fields
        .groupBy(evidence_unique_cols)
        .agg(
            F.collect_set(F.col('biosample')).alias('biosamples'),
            F.collect_set(F.col('study')).alias('studies'),
        )
        .withColumn('biosamples', F.when(F.size('biosamples') != 0, F.col('biosamples')))
        .withColumn('studies', F.when(F.size('studies') != 0, F.col('studies')))
    )

    # Write output
    logging.info('Evidence strings have been processed. Saving...')
    write_evidence_strings(safety_df, output)
    logging.info(f'{safety_df.count()} evidence of safety liabilities have been saved to {output}. Exiting.')

    return 0

def process_aop(aopwiki: str) -> DataFrame:
    """Loads and processes the AOPWiki input JSON."""

    return (
        spark.read.json(aopwiki)
        # if isHumanApplicable is False, set it to null as the lack of applicability has not been tested - it only shows lack of data
        .withColumn('isHumanApplicable', F.when(F.col('isHumanApplicable') != F.lit(True), F.col("isHumanApplicable")))
        # data bug: some events have the substring "NA" at the start - removal and trim the string
        .withColumn('event', F.trim(F.regexp_replace(F.col('event'), '^NA', '')))
        # data bug: effects.direction need to be in lowercase, this field is an enum
        .withColumn(
            'effects',
            F.transform(
                F.col('effects'),
                lambda x: F.struct(
                    F.lower(x.direction).alias('direction'),
                    x.dosing.alias('dosing'))
            )
        )
        # I need to convert the biosamples array into a struct so that data is parsed the same way as the rest of the sources
        .withColumn('biosample', F.explode_outer('biosamples'))
    )

def process_adverse_events(adverse_events: str) -> DataFrame:
    """
    Loads and processes the adverse events input TSV.

    Ex. input record:
        biologicalSystem | gastrointestinal
        effect           | activation_general
        efoId            | EFO_0009836
        ensemblId        | ENSG00000133019
        pmid             | 23197038
        ref              | Bowes et al. (2012)
        symptom          | bronchoconstriction
        target           | CHRM3
        uberonCode       | UBERON_0005409
        url              | null

    Ex. output record:
        id         | ENSG00000133019
        event      | bronchoconstriction
        datasource | Bowes et al. (2012)
        eventId    | EFO_0009836
        literature | 23197038
        url        | null
        biosample  | {gastrointestinal, UBERON_0005409, null, null, null}
        effects    | [{activation, general}]
    """

    ae_df = (
        spark.read.csv(adverse_events, sep='\t', header=True)
        .select(
            F.col('ensemblId').alias('id'),
            F.col('symptom').alias('event'),
            F.col('efoId').alias('eventId'),
            F.col('ref').alias('datasource'),
            F.col('pmid').alias('literature'),
            'url',
            F.struct(
                F.col('biologicalSystem').alias('tissueLabel'),
                F.col('uberonCode').alias('tissueId'),
                F.lit(None).alias('cellLabel'),
                F.lit(None).alias('cellFormat'),
                F.lit(None).alias('cellId'),
            ).alias('biosample'),
            F.split(F.col('effect'), '_').alias('effects'),
        )
        .withColumn(
            'effects',
            F.struct(
                F.element_at(F.col('effects'), 1).alias('direction'), F.element_at(F.col('effects'), 2).alias('dosing')
            ),
        )
    )

    # Multiple dosing effects need to be grouped in the same record.
    effects_df = ae_df.groupBy('id', 'event', 'datasource').agg(F.collect_set(F.col("effects")).alias("effects"))
    ae_df = ae_df.drop("effects").join(effects_df, on=["id", "event", "datasource"], how="left")

    return ae_df


def process_safety_risk(safety_risk: str) -> DataFrame:
    """
    Loads and processes the safety risk information input TSV.

    Ex. input record:
        biologicalSystem | cardiovascular sy...
        ensemblId        | ENSG00000132155
        event            | heart disease
        eventId          | EFO_0003777
        liability        | Important for the...
        pmid             | 21283106
        ref              | Force et al. (2011)
        target           | RAF1
        uberonId         | UBERON_0004535

    Ex. output record:
        id         | ENSG00000132155
        event      | heart disease
        eventId    | EFO_0003777
        literature | 21283106
        datasource | Force et al. (2011)
        biosample  | {cardiovascular s...
        study      | {Important for th...
    """

    return (
        spark.read.csv(safety_risk, sep='\t', header=True)
        .select(
            F.col('ensemblId').alias('id'),
            'event',
            'eventId',
            F.col('pmid').alias('literature'),
            F.col('ref').alias('datasource'),
            F.struct(
                F.col('biologicalSystem').alias('tissueLabel'),
                F.col('uberonId').alias('tissueId'),
                F.lit(None).alias('cellLabel'),
                F.lit(None).alias('cellFormat'),
                F.lit(None).alias('cellId'),
            ).alias('biosample'),
            F.struct(
                F.col('liability').alias('description'), F.lit(None).alias('name'), F.lit(None).alias('type')
            ).alias('study'),
        )
        .withColumn(
            'event',
            F.when(F.col('datasource').contains('Force'), 'heart disease').when(
                F.col('datasource').contains('Lamore'), 'cardiac arrhythmia'
            ),
        )
        .withColumn(
            'eventId',
            F.when(F.col('datasource').contains('Force'), 'EFO_0003777').when(
                F.col('datasource').contains('Lamore'), 'EFO_0004269'
            ),
        )
    )


def process_toxcast(toxcast: str) -> DataFrame:
    """
    Loads and processes the ToxCast input table.

    Ex. input record:
        assay_component_endpoint_name | ACEA_ER_80hr
        assay_component_desc          | ACEA_ER_80hr, is ...
        biological_process_target     | cell proliferation
        tissue                        | null
        cell_format                   | cell line
        cell_short_name               | T47D
        assay_format_type             | cell-based
        official_symbol               | ESR1
        eventId                       | null

    Ex. output record:
     targetFromSourceId | ESR1
    event              | cell proliferation
    eventId            | null
    biosample          | {null, null, T47D...
    datasource         | ToxCast
    url                | https://www.epa.g...
    study              | {ACEA_ER_80hr, AC...
    """

    return spark.read.csv(toxcast, sep='\t', header=True).select(
        F.trim(F.col('official_symbol')).alias('targetFromSourceId'),
        F.col('biological_process_target').alias('event'),
        'eventId',
        F.struct(
            F.col('tissue').alias('tissueLabel'),
            F.lit(None).alias('tissueId'),
            F.col('cell_short_name').alias('cellLabel'),
            F.col('cell_format').alias('cellFormat'),
            F.lit(None).alias('cellId'),
        ).alias('biosample'),
        F.lit('ToxCast').alias('datasource'),
        F.lit('https://www.epa.gov/chemical-research/exploring-toxcast-data-downloadable-data').alias('url'),
        F.struct(
            F.col('assay_component_endpoint_name').alias('name'),
            F.col('assay_component_desc').alias('description'),
            F.col('assay_format_type').alias('type'),
        ).alias('study'),
    )

def get_parser():
    """Get parser object for script TargetSafety.py."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--toxcast',
        help='Input table containing biological processes associated with relevant targets that have been observed in toxicity assays.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--adverse_events',
        help='Input TSV containing adverse events associated with targets that have been collected from relevant publications. Fetched from https://raw.githubusercontent.com/opentargets/curation/master/target_safety/adverse_effects.tsv.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--safety_risk', help='Input TSV containing cardiovascular safety liabilities associated with targets that have been collected from relevant publications. Fetched from https://raw.githubusercontent.com/opentargets/curation/master/target_safety/safety_risks.tsv.', type=str, required=True
    )
    parser.add_argument(
        '--aopwiki', help='Input JSON containing targets implicated in adverse outcomes as reported by the AOPWiki. Parsed from their source XML data.', type=str, required=True
    )
    parser.add_argument(
        '--output', help='Output gzipped json file following the target safety liabilities data model.', type=str, required=True
    )
    parser.add_argument(
        '--log_file',
        help='Destination of the logs generated by this script. Defaults to None',
        type=str,
        nargs='?',
        default=None
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    main(
        toxcast=args.toxcast,
        adverse_events=args.adverse_events,
        safety_risk=args.safety_risk,
        aopwiki=args.aopwiki,
        output=args.output
    )
