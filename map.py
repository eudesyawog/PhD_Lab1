import glob
import os
import re
import rasterio 
from pprint import pprint
from datetime import datetime
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

class Classifier :
    def __init__(self,inPath,ground_truth,niter=5,DateTime=None):
        """
        """
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

        lstSpectral = [os.path.join(self._inPath,"GAPF",File) for File in os.listdir(os.path.join(self._inPath,"GAPF")) if File.endswith("GAPF.tif")]
        lstSpectral.sort()
        lstSpectral.extend([os.path.join(self._inPath,"INDICES",File) for File in os.listdir(os.path.join(self._inPath,"INDICES")) if File.endswith("GAPF.tif")])

        with rasterio.open(lstSpectral[0]) as ds:
            self._profile = ds.profile
            self._width = ds.width
            self._height = ds.height
        self._profile.update(count=1,dtype=rasterio.int16)

        # EMP
        # self._emp99_data = os.path.join(self._outPath,"DATA","emp99_data.npy")
        
        # Spectral data
        self._spectral_data = os.path.join(self._outPath,"DATA","spectral_data.npy")
        
        # EMP99 + Spectral data
        self._total99_data = os.path.join(self._outPath,"DATA","total99_data.npy")
        
    def classify (self):
        # emp99_data = np.load(self._emp99_data)
        spectral_data = np.load(self._spectral_data)
        total99_data = np.load(self._total99_data)

        if not os.path.isdir(os.path.join(self._outPath,"MAP_%s"%self._datetime)):
            os.makedirs(os.path.join(self._outPath,"MAP_%s"%self._datetime))
    
        for i in range(self._niter,self._niter+1):
            # Spectral Bands
            spectral_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1))) 
            spectral_predict = spectral_model.predict(spectral_data)
            spectral_map = spectral_predict.reshape((self._width,self._height))
            with rasterio.open(os.path.join(self._outPath,"MAP_%s"%self._datetime,'spectral_map_iteration%s.tif'%(i+1)),'w',**self._profile) as outDS :
                outDS.write(spectral_map,1)
            print ("Map produced with Spectral Features !")

            # EMP99
            # emp99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1)))
            # emp99_predict = emp99_model.predict(emp99_data)
            # emp99_map = emp99_predict.reshape((self._width,self._height))
            # with rasterio.open(os.path.join(self._outPath,"MAP_%s"%self._datetime,'emp99_map_iteration%s.tif'%(i+1)),'w',**self._profile) as outDS :
            #     outDS.write(emp99_map,1)
            # print ("Map produced with EMP 99 Features !")

            # EMP99 + Spectral Bands
            total99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1)))
            total99_predict = total99_model.predict(total99_data)
            total99_map = total99_predict.reshape((self._width,self._height))
            with rasterio.open(os.path.join(self._outPath,"MAP_%s"%self._datetime,'total99_map_iteration%s.tif'%(i+1)),'w',**self._profile) as outDS :
                outDS.write(total99_map,1)
            print ("Map produced with EMP 99 + Spectral Features !")

if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # Classification
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/REUNION_GT_SAMPLES.shp"
    CO = Classifier(inPath,ground_truth,niter=3,DateTime="0201_1101")
    CO.classify()

    # ========
    # DORDOGNE
    # ========

    # Classification
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/DORDOGNE/SOURCE_VECTOR/DORDOGNE_GT_SAMPLES_BUF-10_NOROADS.shp"
    CO = Classifier(inPath,ground_truth,niter=3,DateTime="0201_1102")
    CO.classify()