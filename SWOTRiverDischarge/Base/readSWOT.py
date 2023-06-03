import logging
import os
import glob
import netCDF4 as nc
import geopandas as gpd

class SWOTDataReader(object):
    
    def __init__(self, workspace):
        logging.info("[INFO]Reading data...")
        self.workspace = workspace
        self.shapeFiles = glob.glob(os.path.join(self.workspace, "*.shp"))
        self.ncFiles = glob.glob(os.path.join(self.workspace, "*.nc"))

    def __del__(self):
        logging.info("[INFO]Reading data...")

    def read_shp(self):
        shpFile = self.shapeFiles[0]
        gdf = gpd.read_file(shpFile)
        gdf = gdf[(gdf["wse"] > 0) & (gdf["width"] > 0)]

        shpFile = self.shapeFiles[1]
        gdf = gpd.read_file(shpFile)
        gdf = gdf[(gdf["wse"] > 0) & (gdf["width"] > 0)]
        # filter

        
def main():
    
    log = logging.getLogger()
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    
    fileName = r"C:\Users\lwx\source\repos\RiverDishcharge\RiverDishcharge\Data\SWOTSampleData"
    reader = SWOTDataReader(fileName)
    reader.read_shp()
if __name__ == '__main__':
    main()
