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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, f1_score

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
            
            prop = np.cumsum(val_propres) * 100/tot
            outdic.setdefault('Time series',[]).append(os.path.basename(inFile).split('_',1)[0])
            outdic.setdefault('90% Variance',[]).append(np.where(prop>=90)[0][0]+1)
            outdic.setdefault('95% Variance',[]).append(np.where(prop>=95)[0][0]+1)
            outdic.setdefault('99% Variance',[]).append(np.where(prop>=99)[0][0]+1)
                
            # for i,j,k in zip(val_propres,np.cumsum(val_propres),range(self._count)) :
            #     outdic.setdefault('Time series',[]).append(os.path.basename(inFile).split('_',1)[0])
            #     outdic.setdefault('Component',[]).append(k+1)
            #     outdic.setdefault('Standard Deviation',[]).append(i**0.5)
            #     outdic.setdefault('Proportion of Variance (%)',[]).append(np.round(i*100 /tot,2))
            #     outdic.setdefault('Cumulative Proportion (%)',[]).append(np.round(j*100/tot,2))

            self._recover = False

        outdf = pd.DataFrame.from_dict(outdic)
        outdf.to_csv(outCSV , index=False)

    def save_pc (self, pvar=90):
        """
        Save Principal Components that keep (percent) of variance
        """

        for inFile in self._lstFiles:
            print (inFile, "\t normalization : %s"%self._norm)
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
            ncomp = np.where(prop>=pvar/100)[0][0]+1

            model2 = PCA(n_components=ncomp)
            Z = model2.fit_transform(X)
            Z = Z.reshape(self._rows, self._cols,ncomp)
            
            outFile = os.path.join(self._outPath,os.path.basename(inFile).replace("CONCAT_S2_GAPF.tif","PCA.tif"))
            self._profile.update(nodata=None, count = ncomp, dtype=rasterio.dtypes.get_minimum_dtype(Z))

            with rasterio.open(outFile, 'w', **self._profile) as ds :
                for i in range(ncomp):
                    ds.write(Z[:,:,i].astype(rasterio.dtypes.get_minimum_dtype(Z)),i+1)

            self._recover = False

class Morpho_Operation :
    """
    Performs morphological operations on a grayscale input image
    """
    def __init__(self, inPath, NUM_WORKERS=5):
        self._inPath = inPath
        self._outPath = os.path.dirname(inPath)+os.sep+"EMP"
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
                    Command = ['otbcli_GrayScaleMorphologicalOperation','-in',File,'-channel',str(i),]
                    Command += ['-structype','ball','-structype.ball.xradius', str(j),'-structype.ball.yradius',str(j)]
                    Command += ['-filter', str(k), '-out',os.path.join(tmp_folder,tmp_file%(i,k,j)),'-ram','8192']
                    # print (Command)
                    subprocess.call(Command,shell=False)
        return File
    
    def _process_emp (self,Folder):
        
        lstFiles = glob.glob(os.path.join(self._outPath,Folder)+os.sep+"*.tif")
        lstFiles.sort()
        outFile = os.path.basename(lstFiles[0]).split('_')[0] +'_'+ os.path.basename(lstFiles[0]).split('_')[1] +"_EMP.tif"
        Command = ['otbcli_ConcatenateImages','-il']
        Command.extend(lstFiles)
        Command += ['-out',os.path.join(self._outPath,outFile),'-ram','8192']
        subprocess.call(Command,shell=False)

        return os.path.join(self._outPath,Folder)

    def create_mp(self):
        lstFiles = glob.glob(self._inPath+os.sep+"*.tif")
        lstFiles.sort()
        pool = Pool(processes=self._NUM_WORKERS)
        start_time = time.time()
        results = pool.imap(self._process_mp, lstFiles)
        for result in results :
            print (result)
        end_time = time.time()
        print("Time for process : %ssecs" % (end_time - start_time))
    
    def create_emp(self):
        lstFolders = [Folder for Folder in os.listdir(self._outPath) if os.path.isdir(os.path.join(self._outPath,Folder))]
        lstFolders.sort()
        pool = Pool(processes=self._NUM_WORKERS)
        start_time = time.time()
        results = pool.imap(self._process_emp, lstFolders)
        for result in results :
            print (result)
        end_time = time.time()
        print("Time for process : %ssecs" % (end_time - start_time))

class Classifier :
    def __init__(self,inPath,ground_truth):
        self._inPath = inPath
        self._gt = ground_truth
        self._outPath = os.path.join(inPath,"CLASSIF")
        if not os.path.isdir(self._outPath):
            os.makedirs(self._outPath)
        self._datetime = datetime.now().strftime("%m%d_%H%M")
        self._prepare_classif()
    
    def _prepare_classif(self) :

        if not os.path.isdir(os.path.join(self._outPath,"DATA")):
            os.makedirs(os.path.join(self._outPath,"DATA"))
        
        lstSpectral = [os.path.join(self._inPath,"GAPF",File) for File in os.listdir(os.path.join(self._inPath,"GAPF")) if File.endswith("GAPF.tif")]
        lstSpectral.sort()
        lstSpectral.extend([os.path.join(self._inPath,"INDICES",File) for File in os.listdir(os.path.join(self._inPath,"INDICES")) if File.endswith("GAPF.tif")])

        ds = gdal.Open(lstSpectral[0])
        self._geoT = ds.GetGeoTransform()
        self._proj = ds.GetProjection()
        self._xsize = ds.RasterXSize
        self._ysize = ds.RasterYSize
        ds = None
        
        mem_drv = gdal.GetDriverByName("MEM")
        gt_shp = ogr.Open(self._gt)
        gt_layer = gt_shp.GetLayer()

        dest1 = mem_drv.Create('', self._xsize, self._ysize, 1, gdal.GDT_Byte)
        dest1.SetGeoTransform(self._geoT)
        dest1.SetProjection(self._proj)
        gdal.RasterizeLayer(dest1, [1], gt_layer, options=["ATTRIBUTE=Code2"]) #Code2
        gt_rst = dest1.GetRasterBand(1).ReadAsArray()

        dest2 =  mem_drv.Create('', self._xsize, self._ysize, 1, gdal.GDT_UInt16)
        dest2.SetGeoTransform(self._geoT)
        dest2.SetProjection(self._proj)
        gdal.RasterizeLayer(dest2, [1], gt_layer, options=["ATTRIBUTE=ID"]) #ID
        ID_rst  = dest2.GetRasterBand(1).ReadAsArray()

        gt_shp = None
        gt_layer = None
        dest1 = None
        dest2 = None
        mem_drv = None

        self._gt_indices = np.nonzero(gt_rst)
        self._gt_labels = gt_rst[self._gt_indices]
        self._gt_ID = ID_rst[self._gt_indices]

        # EMP
        self._emp_data = os.path.join(self._outPath,"DATA","emp_data.npy")
        if not os.path.exists(self._emp_data) :
            lstEMP = glob.glob(os.path.join(self._inPath,"EMP")+os.sep+"*.tif")
            lstEMP.sort()
            count = 0
            for File in lstEMP :
                with rasterio.open(File) as ds :
                    count += ds.count
            emp_samples = np.empty((self._gt_labels.shape[0],count))
            count = 0
            for File in lstEMP :
                with rasterio.open(File) as ds :
                    for j in range(ds.count):
                        emp_samples[:,j+count] = ds.read(j+1)[self._gt_indices]
                count += ds.count
            np.save(self._emp_data,emp_samples)
        
        # Spectral data
        self._spectral_data = os.path.join(self._outPath,"DATA","spectral_data.npy")
        if not os.path.exists(self._spectral_data) :
            count = 0
            for File in lstSpectral :
                with rasterio.open(File) as ds :
                    count += ds.count
            spectral_samples = np.empty((self._gt_labels.shape[0],count))
            count = 0
            for File in lstSpectral :
                with rasterio.open(File) as ds :
                    for j in range(ds.count):
                        spectral_samples[:,j+count] = ds.read(j+1)[self._gt_indices]
                count += ds.count
            np.save(self._spectral_data,spectral_samples)

    def classify (self):
        
        emp_data = np.load(self._emp_data)
        spectral_data = np.load(self._spectral_data)

        gt = gpd.read_file(self._gt)
        lstClass = list(set(gt["Code2"].tolist())) #Code2

        outCSV = os.path.join(self._outPath,"classification_results_%s.csv"%self._datetime)
        outdic = {}

        for i in range(5): #5 iterations
            print ("Iteration %s"%(i+1))
            
            train_ID = []
            test_ID = []
            for c in lstClass :
                lstID = gt["ID"][gt["Code2"] == c].tolist() #Code2
                random.shuffle(lstID)
                random.shuffle(lstID)
                train_ID.extend(lstID[:round(len(lstID)*0.7)]) #Train Test proportion 70%
                test_ID.extend(lstID[round(len(lstID)*0.7):])
            
            if not os.path.isdir(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime)):
                os.makedirs(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime))
            
            train_df = gt[gt["ID"].isin(train_ID)]
            train_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1)))
            
            test_df = gt[gt["ID"].isin(test_ID)]
            test_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1)))

            if not os.path.isdir(os.path.join(self._outPath,"MODELS_%s"%self._datetime)):
                os.makedirs(os.path.join(self._outPath,"MODELS_%s"%self._datetime))

            # Training
            train_ix = np.where(np.isin(self._gt_ID, train_ID))
            train_labels = self._gt_labels[train_ix]

            # EMP Model
            emp_train_samples = emp_data[train_ix]
            emp_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
            emp_model.fit(emp_train_samples,train_labels)
            joblib.dump(emp_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp_model_iteration%s.pkl'%(i+1)))
            # emp_model = joblib.load(os.path.join(self._outPath,"MODELS",'emp_model_iteration%s.pkl'%(i+1)))
            
            # Spectral Model
            spectral_train_samples = spectral_data[train_ix]
            spectral_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
            spectral_model.fit(spectral_train_samples,train_labels)
            joblib.dump(spectral_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1)))
            # spectral_model = joblib.load(os.path.join(self._outPath,"MODELS",'spectral_model_iteration%s.pkl'%(i+1)))

            # Testing
            test_ix = np.where(np.isin(self._gt_ID, test_ID))
            test_labels =  self._gt_labels[test_ix]

            # Predict & Metrics EMP
            emp_test_samples = emp_data[test_ix]
            emp_predict = emp_model.predict(emp_test_samples)
            outdic.setdefault('Iteration',[]).append(str(i+1))
            outdic.setdefault('Input',[]).append("EMP")
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, emp_predict))
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, emp_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, emp_predict,average='weighted'))

            # Predict & Metrics Spectral Data
            spectral_test_samples = spectral_data[test_ix]    
            spectral_predict = spectral_model.predict(spectral_test_samples)
            outdic.setdefault('Iteration',[]).append(str(i+1))
            outdic.setdefault('Input',[]).append("Spectral Bands")
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, spectral_predict))
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, spectral_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, spectral_predict,average='weighted'))
        
        outdf = pd.DataFrame.from_dict(outdic)
        outdf.to_csv(outCSV, index=False)

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
    
    # acp = ACP(lstFiles)
    # acp.check_pc(norm=True)
    # acp.check_pc(norm=False)
    # acp.save_pc()

    # EMP
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/PCA"
    # morpho = Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    # Classification
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/BD_GABIR_2017_v3_ID.shp"

    CO = Classifier(inPath,ground_truth)
    CO.classify()


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
    
    # acp = ACP(lstFiles)
    # acp.check_pc(norm=True)
    # acp.check_pc(norm=False)
    # acp.save_pc()

    # EMP
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/PCA"
    # morpho = Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    # Classification
