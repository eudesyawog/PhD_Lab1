#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:43:19 2018

@author: je
"""

import glob
import os
import re
import rasterio 
from pprint import pprint
import numpy as np
import time

class Spectral_Indices :
    """
    Compute spectral indices from Sentinel-2 Time Series :
        - NDVI (Normalized Difference Vegetation Index) -->  nir (b8) - red (b4) / nir (b8) + red (b4)
    """
    def __init__(self, inPath, lstSI = ["NDVI"]):
        self._inPath = inPath
        self._lstSI = lstSI
    
    def compute(self):
        lstFiles = glob.glob(self._inPath+os.sep+"*GAPF.tif")

        for File in lstFiles :
            # B8 ~ NIR
            if re.match("{}_(.*)".format("B8"),os.path.basename(File)):
                nir_concat = File
        for File in lstFiles :
            # B4 ~ RED
            if re.match("{}_(.*)".format("B4"),os.path.basename(File)):
                red_concat = File

        with rasterio.open(lstFiles[0]) as ds :
            nb = ds.count
            xsize = ds.width
            ysize = ds.height
            profile = ds.profile
            profile.update(nodata=np.nan,dtype=rasterio.float64)
            print (profile)

        if "NDVI" in self._lstSI :
            outPath = os.path.dirname(self._inPath) + os.sep + "NDVI"
            if not os.path.isdir(outPath):
                os.makedirs(outPath)
            outFile = "NDVI_"+os.path.basename(lstFiles[0]).split('_',1)[1]

            ndvi_array = np.empty((ysize,xsize,nb))
            for i in range (nb):
                with rasterio.open(nir_concat) as ds :
                    nir = ds.read(i+1)
                    nir = np.where(nir==-10000,np.nan,nir)
                with rasterio.open(red_concat) as ds :
                    red = ds.read(i+1)
                    red = np.where(red==-10000,np.nan,red)

                ndvi_array[:,:,i] = np.where((nir+red)>0,(nir-red)/(nir+red),0)
            
            with rasterio.open(os.path.join(outPath,outFile),'w', **profile) as outDS :
                outDS.write(ndvi_array,nb)
        
        
if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # NDVI Computation

    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF"
    SIObject = Spectral_Indices(inPath)
    SIObject.compute()

    # PCA

    # EMP

    # Classification


    # ========
    # DORDOGNE
    # ========

    # NDVI Computation

    # PCA

    # EMP

    # Classification
