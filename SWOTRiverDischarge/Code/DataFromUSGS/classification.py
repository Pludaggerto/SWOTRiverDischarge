NWIS = r"C:\Users\lwx\Desktop\geobam\field_measurements.csv"
NHD  = r"C:\Users\lwx\Desktop\geobam\NHD_join_table.csv"

import pandas as pd

NHD_df  = pd.read_csv(NHD)
NWIS_df = pd.read_csv(NWIS)

print(NHD_df)