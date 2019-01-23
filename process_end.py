import glob
import os
import re
import rasterio
import gdal
import ogr 
from pprint import pprint
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing, disk,  

class Geodesic_Morpho_Operation :
    """
    Performs geoedesic morphological operations on a grayscale input image
    """

    def __init__(self, inPath, NUM_WORKERS=5):
        """
        """
        self._inPath = inPath
        self._outPath = os.path.dirname(inPath)+os.sep+"EMP_Geodesic_%s"%os.path.basename(inPath).split('_')[-1]
        if not os.path.isdir(self._outPath):
            os.makedirs(self._outPath)
        self._NUM_WORKERS = NUM_WORKERS
    
    def _process_mp (self,File):
        
        tmp_folder = self._outPath+os.sep+os.path.basename(File).split('_',1)[0]+"_EMP"
        tmp_file = os.path.basename(File).replace(".tif","_C%s_%s_%s.tif")
        if not os.path.isdir(tmp_folder):
            os.makedirs(tmp_folder)
        
        with rasterio.open(File) as ds:
            count = ds.count       
            for i in range(1,count+1):
                for j in range(3,8,2):
                for k in ("dilate","erode","opening","closing"):
                    selem = disk(j)
                    eroded = erosion(ds.read(i), selem)

class Classifier :
    def __init__(self,inPath,ground_truth,niter=5,DateTime=None):
        """
        """

if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # ========
    # DORDOGNE
    # ========