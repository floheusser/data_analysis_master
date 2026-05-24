# Original notebook: 04_collect_cjeu_cases_via_eurlex.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1 ---
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import math
import hashlib
import os

EURLEX_WS_URL = 'https://eur-lex.europa.eu/EURLexWebService'
WS_USERNAME = 'n00mbdci'
WS_PASSWORD = 'vJgHkuHuFU3'

DEFAULT_PAGE_SIZE = 100
FORCE_REFRESH = False
MAX_PAGES = None  # Set to e.g. 3 for testing

NS = {'s': 'http://eur-lex.europa.eu/search'}
SNS = '{http://eur-lex.europa.eu/search}'

EXPERT_QUERY = '''
SELECT CELLAR_ID, DN, TI, CI, LB_ART, TI_SHORT, FM, CT, DD, PROC_NUM, FM, TT, I1, I2, ECLI
 WHERE
(FM ~ "Judgment" OR "Opinion" OR "Order")
AND
(CT = "CONC" OR "ENTR" OR "POSI")
'''

CACHE_DIR = 'data/raw/eurlex_cjeu_ws'
os.makedirs(CACHE_DIR, exist_ok=True)

print('Configuration geladen.')
print(f'FORCE_REFRESH={FORCE_REFRESH}, MAX_PAGES={MAX_PAGES}, DEFAULT_PAGE_SIZE={DEFAULT_PAGE_SIZE}')

# --- Cell 2 ---
def get_query_cache_key(expert_query):
    return hashlib.md5(expert_query.strip().encode('utf-8')).hexdigest()[:12]


def get_ws_cache_path(query_key, page):
    return os.path.join(CACHE_DIR, f'{query_key}_page_{page:03d}.xml')


def build_soap_request(expert_query, page, page_size):
    return f'''
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:sear="http://eur-lex.europa.eu/search">
  <soap:Header>
    <wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" soap:mustUnderstand="true">
      <wsse:UsernameToken xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd" wsu:Id="UsernameToken-1">
        <wsse:Username>{WS_USERNAME}</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">{WS_PASSWORD}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
  </soap:Header>
  <soap:Body>
    <sear:searchRequest>
      <sear:expertQuery><![CDATA[{expert_query}]]></sear:expertQuery>
      <sear:page>{page}</sear:page>
      <sear:pageSize>{page_size}</sear:pageSize>
      <sear:searchLanguage>en</sear:searchLanguage>
      <sear:showDocumentsAvailableIn>en,de</sear:showDocumentsAvailableIn>
    </sear:searchRequest>
  </soap:Body>
</soap:Envelope>'''


def eurlex_ws_search_page(expert_query, page=1, page_size=DEFAULT_PAGE_SIZE, use_cache=True):
    query_key = get_query_cache_key(expert_query)
    cache_path = get_ws_cache_path(query_key, page)

    if use_cache and not FORCE_REFRESH and os.path.exists(cache_path):
        print(f'  [CACHE] Page {page} aus {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    print(f'  [WS-CALL] Page {page} is being fetched...')
    soap_xml = build_soap_request(expert_query, page, page_size)
    headers = {'Content-Type': 'application/soap+xml; charset=utf-8', 'Accept': 'application/xml'}
    resp = requests.post(EURLEX_WS_URL, data=soap_xml.encode('utf-8'), headers=headers, timeout=60)
    resp.raise_for_status()

    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    print(f'  [WS-CALL] Page {page} saved: {cache_path}')
    return resp.text


print('WS-Functions definiert.')

# --- Cell 3 ---
def parse_search_response(xml_text):
    root = ET.fromstring(xml_text)

    def get_val(tag):
        el = root.find(f'.//{SNS}{tag}')
        return int(el.text) if el is not None and el.text else 0

    numhits = get_val('numhits')
    totalhits = get_val('totalhits')
    page = get_val('page')
    results = list(root.iter(f'{SNS}result'))

    return {
        'numhits': numhits,
        'totalhits': totalhits,
        'page': page,
        'results': results,
    }


print('Parsing function defined.')

# --- Cell 4 ---
def get_first_text(node, xpath):
    if node is None:
        return ''
    el = node.find(xpath, NS)
    return el.text.strip() if el is not None and el.text else ''


def get_cellar_id_from_work(work_node):
    if work_node is None:
        return ''
    for uri in work_node.findall(f'{SNS}URI'):
        type_el = uri.find(f'{SNS}TYPE')
        if type_el is not None and type_el.text == 'cellar':
            return get_first_text(uri, 's:IDENTIFIER')
    return ''


def get_ct_codes(work_node):
    if work_node is None:
        return ''
    codes = []
    for sm in work_node.findall(f'{SNS}RESOURCE_LEGAL_IS_ABOUT_SUBJECT-MATTER'):
        for child in sm:
            id_el = child.find(f'{SNS}IDENTIFIER')
            if id_el is not None and id_el.text:
                code = id_el.text.strip()
                if code not in codes:
                    codes.append(code)
    return '|'.join(codes)


def derive_court_level_from_celex(celex_id):
    if not celex_id:
        return ''
    if 'CJ' in celex_id:
        return 'court_of_justice'
    if 'TJ' in celex_id:
        return 'general_court'
    if 'FJ' in celex_id:
        return 'civil_service_tribunal'
    return ''


def parse_result_to_row(result_node, query_label='', query_text='', ws_page=0, ws_totalhits=0):
    notice = result_node.find(f'.//{SNS}NOTICE')
    if notice is None:
        return None

    work = notice.find(f'{SNS}WORK')
    expression = notice.find(f'{SNS}EXPRESSION')
    manifestation = notice.find(f'{SNS}MANIFESTATION')

    celex_id = get_first_text(work, 's:ID_CELEX/s:VALUE')

    return {
        'cellar_id': get_cellar_id_from_work(work),
        'celex_id': celex_id,
        'title': get_first_text(expression, 's:EXPRESSION_TITLE/s:VALUE'),
        'document_date': get_first_text(work, 's:WORK_DATE_DOCUMENT/s:VALUE'),
        'court_level': derive_court_level_from_celex(celex_id),
        'resource_type_code': get_first_text(work, 's:WORK_HAS_RESOURCE-TYPE/s:IDENTIFIER'),
        'treaty_code': get_first_text(work, 's:RESOURCE_LEGAL_BASED_ON_CONCEPT_TREATY/s:IDENTIFIER'),
        'ct_codes': get_ct_codes(work),
        'case_law_parties_raw': get_first_text(manifestation, 's:MANIFESTATION_CASE-LAW_PARTIES/s:VALUE'),
        '_query_label': query_label,
        '_query_text': query_text,
        '_ws_page': ws_page,
        '_ws_totalhits': ws_totalhits,
    }


print('Parsing-Helper Functions definiert.')

# --- Cell 5 ---
def collect_all_pages_for_query(expert_query, query_label, page_size=DEFAULT_PAGE_SIZE):
    print(f'\n=== Query: {query_label} ===')
    print(f'Query text (start): {expert_query[:80].strip()}...')

    # Step A: Page 1 laden
    xml_text = eurlex_ws_search_page(expert_query, page=1, page_size=page_size)
    parsed = parse_search_response(xml_text)

    totalhits = parsed['totalhits']
    numhits = parsed['numhits']
    print(f'  totalhits={totalhits}, numhits Page 1={numhits}')

    if numhits == 0:
        print('  No hits. Aborting.')
        return []

    # Step B+C: Anzahl Pagen berechnen
    total_pages = math.ceil(totalhits / page_size)
    if MAX_PAGES is not None:
        total_pages = min(total_pages, MAX_PAGES)
    print(f'  Total pages: {total_pages} (MAX_PAGES={MAX_PAGES})')

    all_rows = []

    # Process page 1
    for result in parsed['results']:
        row = parse_result_to_row(result, query_label=query_label, query_text=expert_query,
                                  ws_page=1, ws_totalhits=totalhits)
        if row:
            all_rows.append(row)

    # Step D: Additional pages
    for page_num in range(2, total_pages + 1):
        xml_text = eurlex_ws_search_page(expert_query, page=page_num, page_size=page_size)
        parsed = parse_search_response(xml_text)

        if parsed['numhits'] == 0:
            print(f'  Page {page_num}: numhits=0, Aborting.')
            break

        for result in parsed['results']:
            row = parse_result_to_row(result, query_label=query_label, query_text=expert_query,
                                      ws_page=page_num, ws_totalhits=totalhits)
            if row:
                all_rows.append(row)

        if parsed['numhits'] < page_size:
            print(f'  Page {page_num}: numhits={parsed["numhits"]} < page_size, last page reached.')
            break

    print(f'  Collected: {len(all_rows)} hits before deduplication')
    return all_rows


print('Pagination function defined.')

# --- Cell 6 ---
# Fetch all pages
all_rows = collect_all_pages_for_query(EXPERT_QUERY, query_label='cjeu_competition')

print(f'\nTotal before deduplication: {len(all_rows)} rows')

# Deduplication: first by cellar_id, otherwise by celex_id
seen = set()
deduped_rows = []
for row in all_rows:
    key = row['cellar_id'] if row['cellar_id'] else row['celex_id']
    if key and key not in seen:
        seen.add(key)
        deduped_rows.append(row)
    elif not key:
        deduped_rows.append(row)

print(f'After deduplication: {len(deduped_rows)} rows')

columns = ['cellar_id', 'celex_id', 'title', 'document_date', 'court_level',
           'resource_type_code', 'treaty_code', 'ct_codes', 'case_law_parties_raw']

df = pd.DataFrame(deduped_rows)
df = df[columns]

output_path = 'data/processed/cjeu_cases.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f'Saved: {len(df)} rows → {output_path}')
df.head()

