import zipfile
import pandas as pd
import os
import argparse

def rectify_sub(inputpath, outpath):
    data = pd.read_csv(inputpath)
    data.uuid = data.uuid.apply(lambda x: x[2:-2])
    data.to_csv("results.csv", index=None)
    with zipfile.ZipFile(outpath, 'w') as zf:
        zf.write('results.csv')

    os.remove('results.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_zip", required=True, help="input zip file")
    ap.add_argument("-o", "--output_zip", required=True, help="output zip file")
    args = vars(ap.parse_args())
    inputpath = args["input_zip"]
    outpath = args["output_zip"]
    rectify_sub(inputpath, outpath)