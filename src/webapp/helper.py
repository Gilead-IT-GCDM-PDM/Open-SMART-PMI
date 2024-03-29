import re
import io
import base64
import numpy as np
import pandas as pd
from dash import html


def readexcel(fname, expected_columns=0):
    """
    Autodetects header for an excel file and returns dataframe with contents

    Scans upto 30 rows to detect header
    """
    df = pd.read_excel(fname)
    skip = 0
    cols = df.columns
    count_unnamed = sum([1 for c in cols if 'Unnamed' in c or 'nan' in c])
    while count_unnamed > 3 or len(cols) - count_unnamed < expected_columns:
        cols = [str(x) for x in df.loc[skip].values]
        count_unnamed = sum([1 for c in cols if 'Unnamed' in c or 'nan' in c])
        skip += 1
        if count_unnamed <= 3 and len(cols) - count_unnamed >= expected_columns:
            break

        if skip > 30:
            print(f'Count not find header for file: {fname}')
            break

    df.columns = cols
    df = df[skip:]
    df.reset_index(drop=True, inplace=True)

    df.replace(np.nan, '', inplace=True)
    df.fillna('', inplace=True)
    return df


_INCORRECT_ROWS_FOUND = '%s rows found for: %s'
_LENGTH_MISMATCH = 'Output length = %d, Expected = %d'
_MISMATCH_FOR_ATTR = '%s: %s vs %s (expected). Row=%s'


def compare_output(actual, expected, cols=None, match_cols=[]):
    if cols is None:
        cols = expected.columns

    for col in cols:
        expected[col] = expected[col].str.upper()

    for idx, row in expected.iterrows():
        sub_df = actual
        for c in cols:
            sub_df = sub_df[sub_df[c] == row[c]]

        row_str = ', '.join([f'{c}: {row[c]}' for c in cols])
        if len(sub_df) != 1:
            return _INCORRECT_ROWS_FOUND % (len(sub_df), row_str)

        for mc in match_cols:
            val = sub_df[mc].values[0]
            if val != row[mc]:
                return _MISMATCH_FOR_ATTR % (mc, val, row[mc], row_str)
        return 'success'


def _only_alphabets(text):
    if 'WRK' in text:
        return 'WRK-01897'
    without_numbers = re.sub('[^a-zA-Z-&/ ()]', '', text)
    return without_numbers.strip()


def parse_contents(content, filename):  # , date):
    df = pd.DataFrame()
    if content is not None:
        content_type, content_string = content.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'xls' in filename:
                # Assume that the user uploaded an excel file
                excel_bytes = io.BytesIO(decoded)
                df = readexcel(excel_bytes)
            elif 'csv' in filename:
                csv_bytes = io.BytesIO(decoded)
                df = pd.read_csv(csv_bytes)
        except Exception as exp:
            print('Exception happened in parse content')
            print(exp)
            # traceback.print_exc()
            df = pd.DataFrame()
    return df


def parse_status(contents, filename, date):
    df = pd.DataFrame()
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'xls' in filename:
                # Assume that the user uploaded an excel file
                excel_bytes = io.BytesIO(decoded)
                print('filame === ', filename)
                df = readexcel(excel_bytes)
                print(df.head())
                print('-'*30)
            elif 'csv' in filename:
                csv_bytes = io.BytesIO(decoded)
                df = pd.read_csv(csv_bytes)
            return html.Label(filename)
        except Exception as exp:
            return html.Label(str(exp))
    return html.Label('Pending')

