"""Verify archived hydrological observations against the public CWC API.

Retains only station, parameter, time and water reading fields.
"""
from pathlib import Path
import json
import hashlib
import requests
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/research/2026-11-16'
BASE = 'https://ffs.india-water.gov.in/iam/api/new-entry-data/specification/'
names = {'018-ubddib':'Sivasagar', '017-UBDDIB':'Bihubar',
         '016-UBDDIB':'Nanglamoraghat', '015-UBDDIB':'Desangpani'}
def expr(field, op, value):
    return {'expression': {'valueIsRelationField':False, 'fieldName':field,
                           'operator':op, 'value':value}}
summaries = []
for code, name in names.items():
    cache = OUT/f'cwc_primary_{name.lower()}_hydrology.json'
    if cache.exists():
        records = json.loads(cache.read_text())
    else:
        legacy = OUT/'cwc_primary_sivasagar_rawcode_response.json'
        if name == 'Sivasagar' and legacy.exists():
            raw = json.loads(legacy.read_text())
        else:
            spec = {'where':expr('id.stationCode','eq',code),
                    'and':expr('id.dataTime','gt','2026-07-01T00:00:00')}
            response = requests.get(BASE, params={'specification':json.dumps(spec, separators=(',',':'))}, timeout=40)
            response.raise_for_status()
            raw = response.json()
        records = [dict(station_code=code, datatype_code=r['id']['datatypeCode'],
                        datetime=r['id']['dataTime'], raw_value=r.get('dataValue'),
                        validated_value=r.get('dataValidatedValue')) for r in raw]
        cache.write_text(json.dumps(records), encoding='utf-8')
        # Replace the earlier raw probe with the same hydrological fields only.
        if name == 'Sivasagar' and legacy.exists():
            legacy.write_text(json.dumps(records), encoding='utf-8')
    assert records, f'No primary observations: {name}'
    primary = pd.DataFrame(records)
    primary['datetime'] = pd.to_datetime(primary.datetime)
    july = primary[(primary.datetime >= '2026-07-01') & (primary.datetime < '2026-08-01')].copy()
    july['value'] = july.validated_value.combine_first(july.raw_value).astype(float)
    archive = pq.read_table(OUT/'cwc_hourly_latest.parquet',
        filters=[('station_code','=',code.upper()), ('datetime','>=',pd.Timestamp('2026-07-01')),
                 ('datetime','<',pd.Timestamp('2026-08-01'))]).to_pandas()
    paired = july.merge(archive, on=['datetime','datatype_code'], suffixes=('_primary','_archive'))
    assert not july.duplicated(['datetime','datatype_code']).any()
    assert not archive.duplicated(['datetime','datatype_code']).any()
    paired['difference'] = paired.value_primary-paired.value_archive
    paired[['datetime','datatype_code','value_primary','value_archive','difference']].to_csv(
        OUT/f'cwc_primary_{name.lower()}_july_comparison.csv', index=False)
    report = dict(station=name, raw_code=code, primary_july_rows=len(july),
                  archive_july_rows=len(archive), paired_rows=len(paired),
                  codes=july.datatype_code.value_counts().to_dict(),
                  max_absolute_difference=float(paired.difference.abs().max()),
                  primary_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    summaries.append(report)
    print(json.dumps(report), flush=True)
metadata = dict(endpoint=BASE, retrieved_utc=str(pd.Timestamp.now(tz='UTC')),
                notes='Public CWC API readings; validated value preferred when present. Direct source agreement does not establish gauge/DEM datum alignment or discharge availability.',
                stations=summaries)
(OUT/'cwc_primary_verification.json').write_text(json.dumps(metadata, indent=2))
