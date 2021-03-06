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
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt 

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

    def __init__(self,lstFiles,norm=False,pvar=90) :
        self._lstFiles = lstFiles
        self._pvar = pvar
        self._outPath = os.path.dirname(os.path.dirname(lstFiles[0])) + os.sep + "PCA_%s"%self._pvar
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

    def save_pc (self):
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
                X = None
                for i in range (self._count):
                    if X is None :
                        X = ds.read(i+1).reshape(self._rows*self._cols)
                    else :
                        X = np.column_stack((X,ds.read(i+1).reshape(self._rows*self._cols)))

            if self._norm :
                X = StandardScaler().fit_transform(X)

            model1 = PCA(n_components=self._count)
            model1.fit_transform(X)
            prop = model1.explained_variance_ratio_.cumsum()
            ncomp = np.where(prop>=self._pvar/100)[0][0]+1

            model2 = PCA(n_components=ncomp)
            Z = model2.fit_transform(X)
            Z = Z.reshape(self._rows, self._cols,ncomp)
            
            outFile = os.path.join(self._outPath,os.path.basename(inFile).replace("CONCAT_S2_GAPF.tif","PCA_%s.tif"%self._pvar))
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
        self._outPath = os.path.dirname(inPath)+os.sep+"EMP_%s"%os.path.basename(inPath).split('_')[-1]
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

        outFile = os.path.basename(lstFiles[0]).split('_')[0] +'_'+ os.path.basename(lstFiles[0]).split('_')[1] +"_EMP-PCA.tif"
        Command = ['otbcli_ConcatenateImages','-il']
        Command.extend(lstFiles)
        Command += ['-out',os.path.join(self._outPath,outFile),'-ram','4096']
        subprocess.call(Command,shell=False)

        pcaFiles = glob.glob(self._inPath+os.sep+"*.tif")
        pcaFiles.sort()
        for File in pcaFiles :
            if re.match("{}_(.*)".format(os.path.basename(Folder).split("_")[0]),os.path.basename(File)):
                lstFiles.append(File)
                break

        outFile = os.path.basename(lstFiles[0]).split('_')[0] +'_'+ os.path.basename(lstFiles[0]).split('_')[1] +"_EMP.tif"
        Command = ['otbcli_ConcatenateImages','-il']
        Command.extend(lstFiles)
        Command += ['-out',os.path.join(self._outPath,outFile),'-ram','4096']
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
    def __init__(self,inPath,ground_truth,niter=5,DateTime=None):
        self._inPath = inPath
        self._gt = ground_truth
        self._outPath = os.path.join(inPath,"CLASSIF")
        if not os.path.isdir(self._outPath):
            os.makedirs(self._outPath)
        if DateTime is None :
            self._datetime = datetime.now().strftime("%m%d_%H%M")
        else :
            self._datetime = DateTime
        self._niter = niter
        self._prepare_classif()
    
    def _prepare_classif(self) :

        if not os.path.isdir(os.path.join(self._outPath,"DATA")):
            os.makedirs(os.path.join(self._outPath,"DATA"))

        if not os.path.isfile(os.path.join(self._outPath,"DATA","gt_labels.npy")) and not os.path.isfile(os.path.join(self._outPath,"DATA","gt_id.npy")) :
        
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
            gdal.RasterizeLayer(dest1, [1], gt_layer, options=["ATTRIBUTE=Code"])
            gt_rst = dest1.GetRasterBand(1).ReadAsArray()

            dest2 =  mem_drv.Create('', self._xsize, self._ysize, 1, gdal.GDT_UInt16)
            dest2.SetGeoTransform(self._geoT)
            dest2.SetProjection(self._proj)
            gdal.RasterizeLayer(dest2, [1], gt_layer, options=["ATTRIBUTE=ID"])
            ID_rst  = dest2.GetRasterBand(1).ReadAsArray()

            gt_shp = None
            gt_layer = None
            dest1 = None
            dest2 = None
            mem_drv = None

            self._gt_indices = np.nonzero(gt_rst)
            self._gt_labels = gt_rst[self._gt_indices]
            np.save(os.path.join(self._outPath,"DATA","gt_labels.npy"),self._gt_labels)
            self._gt_ID = ID_rst[self._gt_indices]
            np.save(os.path.join(self._outPath,"DATA","gt_id.npy"),self._gt_ID)
        else:
            self._gt_labels = np.load(os.path.join(self._outPath,"DATA","gt_labels.npy"))
            self._gt_ID = np.load(os.path.join(self._outPath,"DATA","gt_id.npy"))

        # EMP
        self._emp95_data = os.path.join(self._outPath,"DATA","emp95_data.npy")
        if not os.path.isfile(self._emp95_data) :
            lstEMP95 = glob.glob(os.path.join(self._inPath,"EMP_95")+os.sep+"*EMP.tif")
            lstEMP95.sort()
            emp95_samples = None
            for File in lstEMP95 :
                with rasterio.open(File) as ds :
                    for j in range(ds.count) :
                        if emp95_samples is None :
                            emp95_samples = ds.read(j+1)[self._gt_indices]
                        else :
                            emp95_samples = np.column_stack((emp95_samples,ds.read(j+1)[self._gt_indices]))
            np.save(self._emp95_data,emp95_samples)
            emp95_samples = None
        
        self._emp99_data = os.path.join(self._outPath,"DATA","emp99_data.npy")
        if not os.path.isfile(self._emp99_data) :
            lstEMP99 = glob.glob(os.path.join(self._inPath,"EMP_99")+os.sep+"*EMP.tif")
            lstEMP99.sort()
            emp99_samples = None
            for File in lstEMP99 :
                with rasterio.open(File) as ds :
                    for j in range(ds.count) :
                        if emp99_samples is None :
                            emp99_samples = ds.read(j+1)[self._gt_indices]
                        else :
                            emp99_samples = np.column_stack((emp99_samples,ds.read(j+1)[self._gt_indices]))
            np.save(self._emp99_data,emp99_samples)
            emp99_samples = None
        
        # EMP-PCA
        self._emp_pca95_data = os.path.join(self._outPath,"DATA","emp-pca95_data.npy")
        if not os.path.isfile(self._emp_pca95_data) :
            lstEMP_PCA95 = glob.glob(os.path.join(self._inPath,"EMP_95")+os.sep+"*EMP-PCA.tif")
            lstEMP_PCA95.sort()
            emp_pca95_samples = None
            for File in lstEMP_PCA95 :
                with rasterio.open(File) as ds :
                    for j in range(ds.count) :
                        if emp_pca95_samples is None :
                            emp_pca95_samples = ds.read(j+1)[self._gt_indices]
                        else :
                            emp_pca95_samples = np.column_stack((emp_pca95_samples,ds.read(j+1)[self._gt_indices]))
            np.save(self._emp_pca95_data,emp_pca95_samples)
            emp_pca95_samples = None
        
        self._emp_pca99_data = os.path.join(self._outPath,"DATA","emp-pca99_data.npy")
        if not os.path.isfile(self._emp_pca99_data) :
            lstEMP_PCA99 = glob.glob(os.path.join(self._inPath,"EMP_99")+os.sep+"*EMP-PCA.tif")
            lstEMP_PCA99.sort()
            emp_pca99_samples = None
            for File in lstEMP_PCA99 :
                with rasterio.open(File) as ds :
                    for j in range(ds.count) :
                        if emp_pca99_samples is None :
                            emp_pca99_samples = ds.read(j+1)[self._gt_indices]
                        else :
                            emp_pca99_samples = np.column_stack((emp_pca99_samples,ds.read(j+1)[self._gt_indices]))
            np.save(self._emp_pca99_data,emp_pca99_samples)
            emp_pca99_samples = None
        
        # Spectral data
        self._spectral_data = os.path.join(self._outPath,"DATA","spectral_data.npy")
        if not os.path.isfile(self._spectral_data) :
            spectral_samples = None
            for File in lstSpectral :
                with rasterio.open(File) as ds :
                    for j in range(ds.count):
                        if spectral_samples is None :
                            spectral_samples = ds.read(j+1)[self._gt_indices]
                        else :
                            spectral_samples = np.column_stack((spectral_samples,ds.read(j+1)[self._gt_indices]))
            np.save(self._spectral_data,spectral_samples)
            spectral_samples = None

    def classify (self):
        emp95_data = np.load(self._emp95_data)
        emp99_data = np.load(self._emp99_data)
        spectral_data = np.load(self._spectral_data)
        total95_data = np.column_stack((emp95_data,spectral_data))
        total99_data = np.column_stack((emp99_data,spectral_data))
        
        emp_pca95_data = np.load(self._emp_pca95_data)
        total_pca95_data = np.column_stack((emp_pca95_data,spectral_data))
        emp_pca95_data = None

        emp_pca99_data = np.load(self._emp_pca99_data)
        total_pca99_data = np.column_stack((emp_pca99_data,spectral_data))
        emp_pca99_data = None

        gt = gpd.read_file(self._gt)
        dicClass = gt.groupby(["Code"])["Name"].unique().to_dict()
        lstClass = list(dicClass.keys())
        dicCount = gt.groupby(["Code"])["Code"].count().to_dict()

        if not os.path.isdir(os.path.join(self._outPath,"MODELS_%s"%self._datetime)):
            os.makedirs(os.path.join(self._outPath,"MODELS_%s"%self._datetime))
        if not os.path.isdir(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime)):
            os.makedirs(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime))

        outCSV = os.path.join(self._outPath,"results_summary_%s.csv"%self._datetime)
        outCSViter = os.path.join(self._outPath,"results_iterations_%s.csv"%self._datetime)
        outdic = {}

        for i in range(self._niter):
            print ("Iteration %s"%(i+1))
            
            if not os.path.isfile(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1))) and not os.path.isfile(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1))) :
                train_ID = []
                test_ID = []
                for c in lstClass :
                    lstID = gt["ID"][gt["Code"] == c].tolist()
                    random.shuffle(lstID)
                    random.shuffle(lstID)
                    train_ID.extend(lstID[:round(len(lstID)*0.7)]) #Train Test proportion 70%
                    test_ID.extend(lstID[round(len(lstID)*0.7):])
                train_df = gt[gt["ID"].isin(train_ID)]
                train_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1)))
                
                test_df = gt[gt["ID"].isin(test_ID)]
                test_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1)))
            else :
                train_df = gpd.read_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1)))
                train_ID = train_df["ID"].tolist()
                test_df = gpd.read_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1)))
                test_ID = test_df["ID"].tolist()
            
            # Training
            train_ix = np.where(np.isin(self._gt_ID, train_ID))
            train_labels = self._gt_labels[train_ix]

            # EMP Models
            # 95
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp95_model_iteration%s.pkl'%(i+1))):
                emp95_train_samples = emp95_data[train_ix]
                emp95_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                emp95_model.fit(emp95_train_samples,train_labels)
                joblib.dump(emp95_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp95_model_iteration%s.pkl'%(i+1)))
            else:
                emp95_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp95_model_iteration%s.pkl'%(i+1)))
            emp95_train_samples = None
            
            # 99
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1))):
                emp99_train_samples = emp99_data[train_ix]
                emp99_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                emp99_model.fit(emp99_train_samples,train_labels)
                joblib.dump(emp99_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1)))
            else:
                emp99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1)))
            emp99_train_samples = None

            # Spectral Model
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1))):
                spectral_train_samples = spectral_data[train_ix]
                spectral_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                spectral_model.fit(spectral_train_samples,train_labels)
                joblib.dump(spectral_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1)))
            else:
                spectral_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1)))
            spectral_train_samples = None

            # EMP + Spectral Bands Models
            # 95
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp95+spectral_model_iteration%s.pkl'%(i+1))):
                total95_train_samples = total95_data[train_ix]
                total95_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                total95_model.fit(total95_train_samples,train_labels)
                joblib.dump(total95_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp95+spectral_model_iteration%s.pkl'%(i+1)))
            else:
                total95_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp95+spectral_model_iteration%s.pkl'%(i+1)))
            total95_train_samples = None

            # 99
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1))):
                total99_train_samples = total99_data[train_ix]
                total99_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                total99_model.fit(total99_train_samples,train_labels)
                joblib.dump(total99_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1)))
            else:
                total99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1)))
            total99_train_samples = None

            # EMP-PCA + Spectral Bands Models
            # 95
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp-pca95+spectral_model_iteration%s.pkl'%(i+1))):
                total_pca95_train_samples = total_pca95_data[train_ix]
                total_pca95_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                total_pca95_model.fit(total_pca95_train_samples,train_labels)
                joblib.dump(total_pca95_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp-pca95+spectral_model_iteration%s.pkl'%(i+1)))
            else:
                total_pca95_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp-pca95+spectral_model_iteration%s.pkl'%(i+1)))
            total_pca95_train_samples = None
            # 99
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp-pca99+spectral_model_iteration%s.pkl'%(i+1))):
                total_pca99_train_samples = total_pca99_data[train_ix]
                total_pca99_model = RandomForestClassifier(n_estimators=300, verbose=True) #300
                total_pca99_model.fit(total_pca99_train_samples,train_labels)
                joblib.dump(total_pca99_model, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp-pca99+spectral_model_iteration%s.pkl'%(i+1)))
            else:
                total_pca99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp-pca99+spectral_model_iteration%s.pkl'%(i+1)))
            total_pca99_train_samples = None
           
            # Testing
            test_ix = np.where(np.isin(self._gt_ID, test_ID))
            test_labels =  self._gt_labels[test_ix]

            # Predict & Metrics EMP
            # 95
            emp95_test_samples = emp95_data[test_ix]
            emp95_predict = emp95_model.predict(emp95_test_samples) 
            outdic.setdefault('Input',[]).append("EMP95")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, emp95_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, emp95_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, emp95_predict,average='weighted'))
            emp95_per_class = f1_score(test_labels, emp95_predict,average=None)
            # emp95_cm = confusion_matrix(test_labels, emp95_predict)
            emp95_test_samples = None

            # 99
            emp99_test_samples = emp99_data[test_ix]
            emp99_predict = emp99_model.predict(emp99_test_samples) 
            outdic.setdefault('Input',[]).append("EMP99")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, emp99_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, emp99_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, emp99_predict,average='weighted'))
            emp99_per_class = f1_score(test_labels, emp99_predict,average=None)
            # emp99_cm = confusion_matrix(test_labels, emp99_predict)
            emp99_test_samples = None

            # Predict & Metrics Spectral Data
            spectral_test_samples = spectral_data[test_ix]    
            spectral_predict = spectral_model.predict(spectral_test_samples)
            outdic.setdefault('Input',[]).append("Spectral Bands")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, spectral_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, spectral_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, spectral_predict,average='weighted'))
            spectral_per_class = f1_score(test_labels, spectral_predict,average=None)
            # spectral_cm = confusion_matrix(test_labels, spectral_predict)
            spectral_test_samples = None

            # Predict & Metrics EMP + Spectral Data
            # 95
            total95_test_samples = total95_data[test_ix]
            total95_predict = total95_model.predict(total95_test_samples)
            outdic.setdefault('Input',[]).append("EMP95 + Spectral Bands")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, total95_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, total95_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, total95_predict,average='weighted'))
            total95_per_class = f1_score(test_labels, total95_predict,average=None)
            # total95_cm = confusion_matrix(test_labels, total95_predict)
            total95_test_samples = None

            # 99
            total99_test_samples = total99_data[test_ix]
            total99_predict = total99_model.predict(total99_test_samples)
            outdic.setdefault('Input',[]).append("EMP99 + Spectral Bands")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, total99_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, total99_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, total99_predict,average='weighted'))
            total99_per_class = f1_score(test_labels, total99_predict,average=None)
            # total99_cm = confusion_matrix(test_labels, total99_predict)
            total99_test_samples = None

            # Predict & Metrics EMP-PCA + Spectral Data
            # 95
            total_pca95_test_samples = total_pca95_data[test_ix]
            total_pca95_predict = total_pca95_model.predict(total_pca95_test_samples)
            outdic.setdefault('Input',[]).append("EMP-PCA 95 + Spectral Bands")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, total_pca95_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, total_pca95_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, total_pca95_predict,average='weighted'))
            total_pca95_per_class = f1_score(test_labels, total_pca95_predict,average=None)
            # total_pca95_cm = confusion_matrix(test_labels, total_pca95_predict)
            total_pca95_test_samples = None

            # 99
            total_pca99_test_samples = total_pca99_data[test_ix]
            total_pca99_predict = total_pca99_model.predict(total_pca99_test_samples)
            outdic.setdefault('Input',[]).append("EMP-PCA 99 + Spectral Bands")
            outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, total_pca99_predict))
            outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, total_pca99_predict))
            outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, total_pca99_predict,average='weighted'))
            total_pca99_per_class = f1_score(test_labels, total_pca99_predict,average=None)
            # total_pca99_cm = confusion_matrix(test_labels, total_pca99_predict)
            total_pca99_test_samples = None

            print ("EMP95 | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
                round(outdic['Overall Accuracy'][i*7],3),round(outdic['Kappa Coefficient'][i*7],3),round(outdic['F-Measure'][i*7],3)))
            
            print ("EMP99 | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
                round(outdic['Overall Accuracy'][i*7+1],3),round(outdic['Kappa Coefficient'][i*7+1],3),round(outdic['F-Measure'][i*7+1],3)))
            
            print ("Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
                round(outdic['Overall Accuracy'][i*7+2],3),round(outdic['Kappa Coefficient'][i*7+2],3),round(outdic['F-Measure'][i*7+2],3)))
            
            print ("EMP95 + Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
                round(outdic['Overall Accuracy'][i*7+3],3),round(outdic['Kappa Coefficient'][i*7+3],3),round(outdic['F-Measure'][i*7+3],3)))
            
            print ("EMP99 + Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
                round(outdic['Overall Accuracy'][i*7+4],3),round(outdic['Kappa Coefficient'][i*7+4],3),round(outdic['F-Measure'][i*7+4],3)))
            
            print ("EMP-PCA 95 + Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
                round(outdic['Overall Accuracy'][i*7+5],3),round(outdic['Kappa Coefficient'][i*7+5],3),round(outdic['F-Measure'][i*7+5],3)))

            print ("EMP-PCA 99 + Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s\n"%(
                round(outdic['Overall Accuracy'][i*7+6],3),round(outdic['Kappa Coefficient'][i*7+6],3),round(outdic['F-Measure'][i*7+6],3)))
            
        outdf = pd.DataFrame.from_dict(outdic)
        outdf.to_csv(outCSViter, index=False)
        mean_df = outdf.groupby(["Input"])[["Overall Accuracy","Kappa Coefficient","F-Measure"]].mean()
        std_df = outdf.groupby(["Input"])[["Overall Accuracy","Kappa Coefficient","F-Measure"]].std()
        df1 = pd.DataFrame({'Input': ["EMP95", "EMP99", "Spectral Bands", "EMP95 + Spectral Bands","EMP99 + Spectral Bands","EMP-PCA 95 + Spectral Bands","EMP-PCA 99 + Spectral Bands",""],
                    'Overall Accuracy': ["%s +/- %s"%(round(mean_df.loc['EMP95']['Overall Accuracy'],3),round(std_df.loc['EMP95']['Overall Accuracy'],3)),
                                         "%s +/- %s"%(round(mean_df.loc['EMP99']['Overall Accuracy'],3),round(std_df.loc['EMP99']['Overall Accuracy'],3)),
                                         "%s +/- %s"%(round(mean_df.loc['Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['Spectral Bands']['Overall Accuracy'],3)),
                                         "%s +/- %s"%(round(mean_df.loc['EMP95 + Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['EMP95 + Spectral Bands']['Overall Accuracy'],3)),
                                         "%s +/- %s"%(round(mean_df.loc['EMP99 + Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['EMP99 + Spectral Bands']['Overall Accuracy'],3)),
                                         "%s +/- %s"%(round(mean_df.loc['EMP-PCA 95 + Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['EMP-PCA 95 + Spectral Bands']['Overall Accuracy'],3)),
                                         "%s +/- %s"%(round(mean_df.loc['EMP-PCA 99 + Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['EMP-PCA 99 + Spectral Bands']['Overall Accuracy'],3)),
                                         ""],
                                         
                    'Kappa Coefficient' : ["%s +/- %s"%(round(mean_df.loc['EMP95']['Kappa Coefficient'],3),round(std_df.loc['EMP95']['Kappa Coefficient'],3)),
                                           "%s +/- %s"%(round(mean_df.loc['EMP99']['Kappa Coefficient'],3),round(std_df.loc['EMP99']['Kappa Coefficient'],3)),
                                           "%s +/- %s"%(round(mean_df.loc['Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['Spectral Bands']['Kappa Coefficient'],3)),
                                           "%s +/- %s"%(round(mean_df.loc['EMP95 + Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['EMP95 + Spectral Bands']['Kappa Coefficient'],3)),
                                           "%s +/- %s"%(round(mean_df.loc['EMP99 + Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['EMP99 + Spectral Bands']['Kappa Coefficient'],3)),
                                           "%s +/- %s"%(round(mean_df.loc['EMP-PCA 95 + Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['EMP-PCA 95 + Spectral Bands']['Kappa Coefficient'],3)),
                                           "%s +/- %s"%(round(mean_df.loc['EMP-PCA 99 + Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['EMP-PCA 99 + Spectral Bands']['Kappa Coefficient'],3)),
                                           ""],

                    'F-Measure' : ["%s +/- %s"%(round(mean_df.loc['EMP95']['F-Measure'],3),round(std_df.loc['EMP95']['F-Measure'],3)),
                                   "%s +/- %s"%(round(mean_df.loc['EMP99']['F-Measure'],3),round(std_df.loc['EMP99']['F-Measure'],3)),
                                   "%s +/- %s"%(round(mean_df.loc['Spectral Bands']['F-Measure'],3),round(std_df.loc['Spectral Bands']['F-Measure'],3)),
                                   "%s +/- %s"%(round(mean_df.loc['EMP95 + Spectral Bands']['F-Measure'],3),round(std_df.loc['EMP95 + Spectral Bands']['F-Measure'],3)),
                                   "%s +/- %s"%(round(mean_df.loc['EMP99 + Spectral Bands']['F-Measure'],3),round(std_df.loc['EMP99 + Spectral Bands']['F-Measure'],3)),
                                   "%s +/- %s"%(round(mean_df.loc['EMP-PCA 95 + Spectral Bands']['F-Measure'],3),round(std_df.loc['EMP-PCA 95 + Spectral Bands']['F-Measure'],3)),
                                   "%s +/- %s"%(round(mean_df.loc['EMP-PCA 99 + Spectral Bands']['F-Measure'],3),round(std_df.loc['EMP-PCA 99 + Spectral Bands']['F-Measure'],3)),
                                   ""]})
        df1.to_csv(outCSV, index=False)

        dic2 = {}
        dic2.setdefault('Per Class F-Measure',[]).append("EMP95")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(emp95_per_class[f],3))

        dic2.setdefault('Per Class F-Measure',[]).append("EMP99")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(emp99_per_class[f],3))

        dic2.setdefault('Per Class F-Measure',[]).append("Spectral Bands")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(spectral_per_class[f],3))

        dic2.setdefault('Per Class F-Measure',[]).append("EMP95 + Spectral Bands")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(total95_per_class[f],3))

        dic2.setdefault('Per Class F-Measure',[]).append("EMP99 + Spectral Bands")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(total99_per_class[f],3))
        
        dic2.setdefault('Per Class F-Measure',[]).append("EMP-PCA 95 + Spectral Bands")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(total_pca95_per_class[f],3))
        
        dic2.setdefault('Per Class F-Measure',[]).append("EMP-PCA 99 + Spectral Bands")
        for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
            dic2.setdefault('Class %s'%(v),[]).append(round(total_pca99_per_class[f],3))

        dic2.setdefault('Per Class F-Measure',[]).append("")
        for v in list(dicClass.keys()):
            dic2.setdefault('Class %s'%(v),[]).append("")
            
        df2 = pd.DataFrame.from_dict(dic2)
        df2.to_csv(outCSV, mode ='a', index=False)

        df3 = pd.DataFrame({'Class': [str(v) for v in list(dicClass.keys())],
                            'Name' : [dicClass[v][0] for v in list(dicClass.keys())],
                            'Objects': [dicCount[v] for v in list(dicClass.keys())],
                            'Pixels': [np.count_nonzero(self._gt_labels==v) for v in list(dicClass.keys())]
                            })
        df3.to_csv(outCSV, mode='a', index=False)

        dic4 = {}
        dic4.setdefault("Size",[]).append("Total")
        dic4.setdefault("Spectral Bands",[]).append(spectral_data.shape[1])
        dic4.setdefault("PCA95",[]).append(total95_data.shape[1]-total_pca95_data.shape[1])
        dic4.setdefault("PCA99",[]).append(total99_data.shape[1]-total_pca99_data.shape[1])
        dic4.setdefault("EMP95",[]).append(emp95_data.shape[1])
        dic4.setdefault("EMP99",[]).append(emp99_data.shape[1])
        dic4.setdefault("EMP95 + Spectral Bands",[]).append(total95_data.shape[1])
        dic4.setdefault("EMP99 + Spectral Bands",[]).append(total99_data.shape[1])
        dic4.setdefault("EMP-PCA 95 + Spectral Bands",[]).append(total_pca95_data.shape[1])
        dic4.setdefault("EMP-PCA 99 + Spectral Bands",[]).append(total_pca99_data.shape[1])

        df4 = pd.DataFrame.from_dict(dic4)
        df4.to_csv(outCSV, mode ='a', index=False)
        

    # def _plot_cm (self,cm,normalize=True,cmap=plt.cm.plasma) :
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     plt.imshow(cm,interpolation='nearest', cmap=cmap)

if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # NDVI
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF"
    # SIObject = Spectral_Indices(inPath)
    # SIObject.compute()

    # PCA
    lstFiles = ["/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B2_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B3_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B4_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF/B8_REUNION_CONCAT_S2_GAPF.tif",
                "/media/je/SATA_1/Lab1/REUNION/OUTPUT/INDICES/NDVI_REUNION_CONCAT_S2_GAPF.tif"]
    
    acp = ACP(lstFiles,pvar=95)
    # acp.check_pc(norm=True)
    # acp.check_pc(norm=False)
    # acp.save_pc()

    acp = ACP(lstFiles,pvar=99)
    # acp.check_pc(norm=True)
    # acp.check_pc(norm=False)
    # acp.save_pc()

    # EMP
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/PCA_95"
    # morpho = Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/PCA_99"
    # morpho = Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    # Classification
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/REUNION_GT_SAMPLES.shp"
    # CO = Classifier(inPath,ground_truth,DateTime="0112_0908")
    # CO.classify()


    # ========
    # DORDOGNE
    # ========

    # NDVI
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF"
    # SIObject = Spectral_Indices(inPath)
    # SIObject.compute()

    # PCA
    lstFiles = ["/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B2_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B3_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B4_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/GAPF/B8_DORDOGNE_CONCAT_S2_GAPF.tif",
            "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/INDICES/NDVI_DORDOGNE_CONCAT_S2_GAPF.tif"]
    
    acp = ACP(lstFiles,pvar=95)
    # acp.check_pc(norm=True)
    # acp.check_pc(norm=False)
    # acp.save_pc()
    
    acp = ACP(lstFiles,pvar=99)
    # acp.check_pc(norm=True)
    # acp.check_pc(norm=False)
    # acp.save_pc()

    # EMP
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/PCA_95"
    # morpho = Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/PCA_99"
    # morpho = Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    # Classification
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/DORDOGNE/SOURCE_VECTOR/DORDOGNE_GT_SAMPLES_BUF-10_NOROADS.shp"
    CO = Classifier(inPath,ground_truth,DateTime="0114_0924")
    CO.classify()