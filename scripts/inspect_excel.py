import glob
import os

import pandas as pd


data_dir = "data"
files = glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.xls"))

for file_path in files:
    print("\n" + "=" * 100)
    print("FILE:", os.path.basename(file_path))

    workbook = pd.read_excel(file_path, sheet_name=None, header=None)

    for sheet_name, df in workbook.items():
        print("\nSHEET:", sheet_name)
        print("Shape:", df.shape)
        print(df.head(10).to_string())

        print("\nPossible header rows:")
        for i in range(min(10, len(df))):
            values = [str(x) for x in df.iloc[i].tolist() if str(x) != "nan"]
            print(f"Row {i}:", values[:25])
