import geopandas as gpd
import os
import glob
workspace = r"C:\Users\lwx\Desktop\Dishcarge"
jsonFiles = glob.glob(os.path.join(workspace, "*.geojson"))
for json in jsonFiles:
    print(len(gpd.read_file(json)))