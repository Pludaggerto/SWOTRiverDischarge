import os
import glob
from sas7bdat import SAS7BDAT

workspace = r'C:\Users\lwx\Desktop\SelectAttribute'
fileList = glob.glob(os.path.join(workspace, "*.sas7bdat"))

for File in fileList:
    with SAS7BDAT(File) as reader:
        df = reader.to_data_frame()
    print(df.columns)