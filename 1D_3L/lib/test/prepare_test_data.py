import numpy as np
import pandas as pd
import sys
import os

inputs = np.array(['ro_well', 'ro_formation', 'd_well', 'invasion_zone_ro', 'invasion_zone_h'])
outputs = np.array(['A04M01N', 'A10M01N', 'A20M05N', 'A40M05N', 'A80M10N'])


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
    
    # set d_well instead of rad_well
    if 'd_well' not in df.columns:
        df.rename(columns={"rad_well" : "d_well"}, inplace=True)
        df['d_well'] = df['d_well'] * 2

    df = df[np.concatenate((inputs, outputs))]
    df.to_csv("test_data.csv", index=False)


if __name__ == "__main__":
    main()
