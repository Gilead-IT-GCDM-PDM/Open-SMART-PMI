import os
import pandas as pd


def read_file(fname):
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return f.readlines()
    else:
        print(f'{fname} does not exists')
        return []


def add_expected_complexity():
    results_file = '1953998c_Compound_complexity_V001.parallelRF_predictions.txt'
    df = pd.read_csv(results_file, delimiter=' ')
    return df


def parse(fname, expected=None):
    lines = read_file(fname)
    attrs = {}
    parsed = []
    for line in lines:
        tag, result = parse_line(line)
        if tag == 'NEW':
            attrs = {'MOLECULE': result}
            if expected:
                prediction = expected[expected.MOLECULE == result].PREDICTED.values[0] if expected else 0
                attrs['PREDICTED'] = prediction
        elif tag == 'END':
            if len(attrs) == 0:
                raise ValueError("End of molecule unexpected")
            parsed += [attrs]
            attrs = {}
        elif tag == "ATTRS":  #  and "MOE_2D" not in line:
            attrs.update(result)

    df = pd.DataFrame(parsed)
    return df


def parse_line(line):
    if line[:8] == 'MOLECULE':
        return 'NEW', line.split(' ')[1].strip()
    if line[:6] == 'End_Of':
        return 'END', {}

    idx = line.find('des')
    vals = line[idx + 6:].split(' ')
    attrs = {vals[i]: vals[i+1] for i in range(0, len(vals) - 1, 2)}

    return "ATTRS", attrs


def predict_21(df):
    return 1.3745 + 5.2135*df.CHIRAL_ALLATOM_RATIO + 5.2135*df.TOTALATOM_COUNT + 0.0202*df.UNIQUETT


if __name__ == '__main__':
    # filename = '1953998.desc.txt'
    # filename = '307250.desc.txt'
    # filename = '4063744.desc'
    filename = '2381171.desc'
    # expected = add_expected_complexity()
    df = parse(filename) #, expected)
    df.to_csv('output.csv', index=False)


