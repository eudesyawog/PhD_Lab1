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
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

        outPath = os.path.dirname(self._inPath) + os.sep + "INDICES"
        if not os.path.isdir(outPath):
            os.makedirs(outPath)

        for File in lstFiles :
            # B8 ~ NIR
            if re.match("{}_(.*)".format("B8"),os.path.basename(File)):
                nir_concat = File
                break
        for File in lstFiles :
            # B4 ~ RED
            if re.match("{}_(.*)".format("B4"),os.path.basename(File)):
                red_concat = File
                break

        with rasterio.open(lstFiles[0]) as ds :
            nb = ds.count
            xsize = ds.width
            ysize = ds.height
            profile = ds.profile
            profile.update(nodata=np.nan,dtype=rasterio.float64)
        
        if "NDVI" in self._lstSI :
           
            outFile = "NDVI_"+os.path.basename(lstFiles[0]).split('_',1)[1]
            ndvi_array = np.empty((ysize,xsize,nb))

            for i in range (1): # nb
                with rasterio.open(nir_concat) as ds :
                    nir = ds.read(i+1)
                with rasterio.open(red_concat) as ds :
                    red = ds.read(i+1)

                ndvi_array[:,:,i] = np.where((nir+red)==0,0,(nir-red)/(nir+red))
            
            with rasterio.open(os.path.join(outPath,outFile),'w', **profile) as outDS :
                for i in range(nb):
                    outDS.write(ndvi_array[:,:,i],i+1)
class ACP :
    """
    Do Principal Components Analysis on (bands or Spectral indices) Time series
    """

    def __init__(self,lstFiles,norm=False) :
        self._lstFiles = lstFiles
        self._outPath = os.path.dirname(os.path.dirname(lstFiles[0])) + os.sep + "PCA"
        if not os.path.isdir(self._outPath):
            os.makedirs(self._outPath)
        self._norm = norm
        self._recover = True
    
    def check_pc (self,norm=False) :
        """
        Analyze Proportion of Variance and cumulative proportion 
        keeped by principal components
        """
        if (self._norm and not norm) or (not self._norm and norm) :
            self._norm = norm

        outCSV = os.path.join(self._outPath,"pca_norm_%s.csv"%self._norm)
        outdic = {}

        for inFile in self._lstFiles :
            print (inFile, "\t normalization : %s"%self._norm)
            with rasterio.open(inFile) as ds :
                if self._recover :
                    self._rows = ds.height
                    self._cols = ds.width
                    self._count = ds.count
                    self._profile = ds.profile
                X = np.empty((self._rows, self._cols, self._count))
                for i in range (self._count):
                    X[:,:,i] = ds.read(i+1)

            X = X.reshape(self._rows*self._cols,self._count)

            if self._norm :
                X = StandardScaler().fit_transform(X)
            
            df = pd.DataFrame(X)
            cov_matrix = df.cov()
            val_propres,vec_propres = np.linalg.eig(cov_matrix)
            ordre = val_propres.argsort()[::-1]
            val_propres = val_propres[ordre]
            vec_propres = vec_propres[:,ordre]
            tot = sum(val_propres)
            
            # prop = np.cumsum(val_propres) * 100/tot
            # outdic.setdefault('Time series',[]).append(os.path.basename(inFile).split('_',1)[0])
            # outdic.setdefault('90% Variance',[]).append(np.where(prop>=90)[0][0]+1)
            # outdic.setdefault('95% Variance',[]).append(np.where(prop>=95)[0][0]+1)
            # outdic.setdefault('99% Variance',[]).append(np.where(prop>=99)[0][0]+1)
                
            for i,j,k in zip(val_propres,np.cumsum(val_propres),range(self._count)) :
                outdic.setdefault('Time series',[]).append(os.path.basename(inFile).split('_',1)[0])
                outdic.setdefault('Component',[]).append(k+1)
                outdic.setdefault('Standard Deviation',[]).append(i**0.5)
                outdic.setdefault('Proportion of Variance (%)',[]).append(np.round(i*100 /tot,2))
                outdic.setdefault('Cumulative Proportion (%)',[]).append(np.round(j*100/tot,2))

            self._recover = False

        outdf = pd.DataFrame.from_dict(outdic)
        outdf.to_csv(outCSV , index=False)
    
    def _save_pc (self, inFile) :
        
        with rasterio.open(inFile) as ds :
            if self._recover :
                self._rows = ds.height
                self._cols = ds.width
                self._count = ds.count
                self._profile = ds.profile
            X = np.empty((self._rows, self._cols, self._count),dtype=ds.profile['dtype'])
            for i in range (self._count):
                X[:,:,i] = ds.read(i+1)

        X = X.reshape(self._rows*self._cols,self._count)
        if self._norm :
            X = StandardScaler().fit_transform(X)

        model1 = PCA(n_components=self._count)
        model1.fit_transform(X)
        prop = model1.explained_variance_ratio_.cumsum()
        ncomp = np.where(prop>=self._pvar/100)[0][0]+1

        model2 = PCA(n_components=ncomp)
        Z = model2.fit_transform(X)
        Z = Z.reshape(self._rows, self._cols,ncomp)
        
        outFile = os.path.join(self._outPath,os.path.basename(inFile).replace("CONCAT_S2_GAPF.tif","PCA.tif"))
        self._profile.update(nodata=None, count = ncomp, dtype=rasterio.dtypes.get_minimum_dtype(Z))

        with rasterio.open(outFile, 'w', **self._profile) as ds :
            for i in range(ncomp):
                ds.write(Z[:,:,i].astype(rasterio.dtypes.get_minimum_dtype(Z)),i+1)

        self._recover = False
        return (inFile)

    def save_pc (self, pvar=90, NUM_WORKERS=5):
        """
        Save Principal Components that keep (percent) of variance
        """
        self._pvar = pvar
        pool = Pool(processes=NUM_WORKERS)
        start_time = time.time()
        results = pool.imap(self._save_pc, self._lstFiles)
        for result in results :
            print (result)
        end_time = time.time()
        print("Time for process : %ssecs" % (end_time - start_time))
                    
if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # NDVI Computation
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF"
    # SIObject = Spectral_Indices(inPath)
    # SIObject.compute()

    # PCA
    lstFiles = ["/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B2_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B3_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B4_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B8_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/INDICES/NDVI_REUNION_CONCAT_S2_GAPF.tif"]
    
    acp = ACP(lstFiles)
    acp.check_pc(norm=True)
    acp.check_pc(norm=False)
    acp.save_pc()

    # EMP

    # Classification


    # ========
    # DORDOGNE
    # ========

    # NDVI Computation
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF"
    # SIObject = Spectral_Indices(inPath)
    # SIObject.compute()

    # PCA
    lstFiles = ["/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B2_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B3_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B4_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B8_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/INDICES/NDVI_DORDOGNE_CONCAT_S2_GAPF.tif"]
    
    acp = ACP(lstFiles)
    acp.check_pc(norm=True)
    acp.check_pc(norm=False)
    acp.save_pc()

    # EMP

    # Classification
