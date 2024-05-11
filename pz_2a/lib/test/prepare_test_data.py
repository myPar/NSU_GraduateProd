import numpy as np
import pandas as pd
import sys
import os

inputs = np.array(['ro_well', 'ro_formation', 'r_well', 'lambda1'])
outputs = np.array(['rok'])


# concatenate all source files in single dataframe and save it
def main():
    source_dir = "test_data/"
    source_count = len(sys.argv) - 1

    if source_count < 1:    # read all files from base dir
        input_file_names = [source_dir + name for name in os.listdir(source_dir)]
    else:
        input_file_names = [source_dir + name for name in sys.argv[1:]]
    # concatenate dataframes read from input files:
    df = pd.DataFrame({})

    for name in input_file_names:
        df = pd.concat([df, pd.read_csv(name)])

    df = df[np.concatenate((inputs, outputs))]
    df.to_csv("test_data.csv", index=False)


if __name__ == "__main__":
    main()
