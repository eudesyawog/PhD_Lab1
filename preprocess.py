#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:06:54 2018

@author: je
"""
import zipfile
import os
import glob
import re
import subprocess
import fiona
import rasterio
import gdal
import numpy as np
import pandas as pd
from affine import Affine
from itertools import product
import time
import concurrent.futures
from multiprocessing import Pool
from pprint import pprint

class ExtractZip :
    """
    Extract All Images Zip Files
    zipfolder : Folder containing all zip files
    unzipfolder : Folder receiving unzip elements
    """
    def __init__(self, zipfolder, NUM_WORKERS=10) :
        self._zipfolder = zipfolder
        self._unzipfolder = os.path.join(os.path.dirname(self._zipfolder),"OUTPUT","UNZIP")
        if not os.path.isdir(self._unzipfolder):
            os.makedirs(self._unzipfolder)
        self._NUM_WORKERS = NUM_WORKERS

    def process(self, zipFile) :
        zip_ref = zipfile.ZipFile(zipFile, 'r')
        zip_ref.extractall(self._unzipfolder)
        zip_ref.close()
        return (zipFile)
    
    def extract_files(self) :
        lstZip =  glob.glob(self._zipfolder + os.sep + "*.zip")
        lstZip.sort()
        pool = Pool(processes=self._NUM_WORKERS)
        start_time = time.time()
        results = pool.imap(self.process, lstZip)
        for result in results :
            print (result)
        end_time = time.time()
        print("Time for unzipping : %ssecs" % (end_time - start_time))         
    
class Mosaic_Clip :
    """
    Browse unzip folders, Mosaic bands of interest if necessary, Clip them as same as 
    corresponding clouds mask using a vector file extent and Organize images by date
    """
    def __init__(self, unzipfolder, vectorfile, lstBand=["B2","B3","B4","B8","CLM"], product="FRE", NUM_WORKERS=10):
        self._unzipfolder = unzipfolder
        self._vectorfile = vectorfile
        self._lstBand = lstBand
        self._product = product
        self._NUM_WORKERS = NUM_WORKERS
    
    def _clip(self,inFile,outFile,band): 
        if self._recover_metadata :
            vec = fiona.open(self._vectorfile)
            self._ulx = vec.bounds[0]
            self._uly = vec.bounds[3]
            lrx = vec.bounds[2]
            lry = vec.bounds[1]
            vec = None

            with rasterio.open(inFile) as ds:
                originX = ds.affine[2]
                originY = ds.affine[5]
                self._pixelWidth = ds.affine[0]
                self._pixelHeight = ds.affine[4]

                self._xOffset = int((self._ulx- originX) / self._pixelWidth)
                self._yOffset = int((self._uly - originY) / self._pixelHeight)
                self._xsize = int((lrx - self._ulx)/self._pixelWidth)
                self._ysize = int((self._uly - lry)/self._pixelWidth)
        
        if band == "CLM" :
            with rasterio.open(inFile) as ds:
                mask_array = ds.read(1,window=((self._yOffset,self._yOffset+self._ysize),(self._xOffset, self._xOffset+self._xsize)))
                mask_array = np.where(mask_array>0,1,0)

                profile = ds.profile
                profile.update(width=self._xsize,height=self._ysize,transform=(self._ulx,self._pixelWidth,0.,self._uly,0.,self._pixelHeight))
                
            with rasterio.open(outFile,'w', **profile) as outds :
                outds.write(mask_array.astype(rasterio.uint8),1)
        else : 
            with rasterio.open(inFile) as ds:
                band_array = ds.read(1,window=((self._yOffset,self._yOffset+self._ysize),(self._xOffset, self._xOffset+self._xsize)))
                band_array = np.where(band_array!=-10000,band_array+1,-10000) # +1 for flat reflectance

                profile = ds.profile
                profile.update(width=self._xsize,height=self._ysize,nodata=-10000,transform=(self._ulx,self._pixelWidth,0.,self._uly,0.,self._pixelHeight))
            
            with rasterio.open(outFile,'w', **profile) as outds :
                    outds.write(band_array.astype(rasterio.int16),1)
        
    def process(self, date):
        outPath = os.path.dirname(self._unzipfolder) + os.sep + "MOS_CLIP" + os.sep + date
        if not os.path.isdir(outPath):
            os.makedirs(outPath)

        lstFolders = glob.glob(self._unzipfolder + os.sep + "*%s*"%date)
        if len(lstFolders) > 1 :
            # Mosaic before Clip
            self._recover_metadata = True
            for band in self._lstBand :
                # Mosaic 
                if band == "CLM":
                    Command = ["gdalwarp",'-overwrite', "-multi","-ot","Byte"]
                    for d in range (len(lstFolders)):
                        for File in os.listdir(os.path.join(lstFolders[d],"MASKS")):
                            if re.match("(.*)_{}".format("CLM_R1.tif"),File) and File.endswith(".tif") :
                                Command += [os.path.join(lstFolders[d],"MASKS",File)]
                else :
                    Command = ["gdalwarp",'-overwrite', "-srcnodata", "-10000" , "-dstnodata", "-10000", "-multi","-ot","Int16"]
                    for d in range (len(lstFolders)):
                        for File in os.listdir(lstFolders[d]):
                            if re.match("(.*)_{}_{}(.*)".format(self._product,band),File) and File.endswith(".tif") and not File.endswith("B8A.tif"):
                                Command += [os.path.join(lstFolders[d],File)]
                    
                outFile = band+'_'+date+'_MOSAIC_%s'+File.split('_')[0][-2:]+".tif"
                Command += [os.path.join(outPath,outFile%"S")]
                subprocess.call(Command,shell=False) 
            
                # Clip
                self._clip(inFile=os.path.join(outPath,outFile%"S"),outFile=os.path.join(outPath,outFile%"CLIP_S"),band=band)
                self._recover_metadata = False
                os.remove(os.path.join(outPath,outFile%"S"))
                    
        else :
            # Clip only
            self._recover_metadata = True

            hour = os.path.basename(lstFolders[0]).split('_')[1][9:15]
            sensor = os.path.basename(lstFolders[0]).split('_')[0][-2:]
            outName = "{}_"+date+"_"+hour+"_CLIP_S"+sensor+".tif"
            maskName = "CLM_"+date+"_"+hour+"_CLIP_S"+sensor+".tif"
    
            lstFiles = [File for File in os.listdir(lstFolders[0]) if File.endswith('.tif') and not File.endswith('B8A.tif') and File.split('_')[-2]==self._product]
            for band in self._lstBand:
                if band == "CLM" :
                    for File in os.listdir(os.path.join(lstFolders[0],"MASKS")):
                        if re.match("(.*)_{}".format("CLM_R1.tif"),File) and File.endswith(".tif"):
                            self._clip(inFile=os.path.join(lstFolders[0],"MASKS",File), outFile=os.path.join(outPath,maskName), band=band)   
                else :
                    for File in lstFiles :
                        if re.match("(.*)_{}(.*)".format(band),File):
                            self._clip(inFile=os.path.join(lstFolders[0],File), outFile=os.path.join(outPath,outName.format(band)), band=band)
                            
                self._recover_metadata = False

        return lstFolders
    
    def parallelize(self) : 
        
        lstDates = [folder.split('_')[1][:8] for folder in os.listdir(self._unzipfolder) if os.path.isdir(os.path.join(self._unzipfolder,folder))]
        lstDates.sort()
        uniqueDates = []
        for d in range (len(lstDates)):
            if lstDates[d] not in uniqueDates :
                uniqueDates.append(lstDates[d])
        
        pool = Pool(processes=self._NUM_WORKERS)
        start_time = time.time()
        results = pool.imap(self.process, uniqueDates)
        for result in results :
            print (result)
        end_time = time.time()
        print("Time for process : %ssecs" % (end_time - start_time))

class Cloud_Status :
    """
    Browse Mosaic and Clipping Images Folders and Compute Cloud Percent from Cloud Masks
    """
    def __init__(self, inPath):
        self._inPath = inPath
        self._compute()
    
    def _compute(self):

        lstFolders = [folder for folder in os.listdir(self._inPath) if os.path.isdir(os.path.join(self._inPath,folder))]
        lstFolders.sort()

        outCSV = os.path.join(self._inPath,"cloud_status.csv")
        dic = {}

        for d in range(len(lstFolders)):
            for File in os.listdir(os.path.join(self._inPath,lstFolders[d])):
                if re.match("{}_{}_(.*)".format("CLM",lstFolders[d]),File) and File.endswith(".tif"):
                    with rasterio.open(os.path.join(self._inPath,lstFolders[d],File)) as ds :
                        cld_band = ds.read(1)
                        cld_percent = round((np.count_nonzero(cld_band==1) * 100) / (ds.width * ds.height))
                    dic.setdefault('Date',[]).append(lstFolders[d])
                    dic.setdefault('Cloud_Percentage',[]).append(cld_percent)
        
        df = pd.DataFrame.from_dict(dic)
        df.to_csv(outCSV,index=False)

class Concatenate :
    """
    Concatenate images of the times series per band
    """

    def __init__(self, inPath, bands=["B2","B3","B4","B8","CLM"]):
        self._inPath = inPath
        self._bands = bands
    
    def do(self):
        
        dates = [folder for folder in os.listdir(self._inPath) if os.path.isdir(os.path.join(self._inPath,folder))]
        dates.sort()

        outPath = os.path.dirname(self._inPath)+os.sep+"GAPF"
        if not os.path.isdir(outPath):
            os.makedirs(outPath)

        outName = "{}_"+self._inPath.split(os.sep)[-3]+"_CONCAT_S2.tif"

        fichier = open(outPath+os.sep+"dates.txt", "w")
        for m in range(len(dates)):
            fichier.write("%s \n"%dates[m])
        fichier.close()

        for band in self._bands:
            Command = ['otbcli_ConcatenateImages','-il']
            for m in range(len(dates)):
                for File in os.listdir(os.path.join(self._inPath,dates[m])):
                    if re.match("{}_{}(.*)".format(band,dates[m]),File):
                        Command += [os.path.join(self._inPath,dates[m],File)]
            if band != "CLM":
                Command += ['-out',os.path.join(outPath,outName.format(band)),'int16']
            else:
                Command += ['-out',os.path.join(outPath,outName.format(band)),'uint8']
            Command += ['-ram','32768']

            subprocess.call(Command,shell=False)
            
            if band != "CLM" :
                ds = gdal.Open(os.path.join(outPath,outName.format(band)), gdal.GA_Update)
                for i in range(ds.RasterCount):
                    ds.GetRasterBand(i+1).SetNoDataValue(-10000)
                    band = None
                ds = None

class GapFilling :
    """
    Apply temporel gapfilling of Orfeo ToolBox on each band time series
    """
    def __init__(self, inPath, bands=["B2","B3","B4","B8"], method='linear', del_old=False):
        self._inPath = inPath
        self._bands = bands
        self._method = method
        self._del_old = del_old
    
    def do(self):

        lstFiles = [File for File in os.listdir(self._inPath) if File.startswith('B') and File.endswith('.tif')]
        lstFiles.sort()
        ptrn = lstFiles[0].split('_',1)[1]
        
        inName = os.path.join(self._inPath,"{}_"+ptrn)
        outName = os.path.join(self._inPath,"{}_"+ptrn.strip('.tif')+"_GAPF.tif")
        cloudfile = os.path.join(self._inPath,"CLM_"+ptrn)
        datesfile  = os.path.join(self._inPath,"dates.txt")

        for band in self._bands:
            Command = ['otbcli_ImageTimeSeriesGapFilling', '-in', inName.format(band),'-mask', cloudfile, 
                       '-out', outName.format(band), 'int16', '-comp', '1', '-it', self._method,'-id', datesfile, '-ram', '32768']
            subprocess.call(Command,shell=False)
        
        if self._del_old == True:
            os.remove (inName.format(band))

if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # Unzipping
    zipfolder = "/media/je/SATA_1/Lab1/REUNION/IMAGES"
    zipobject = ExtractZip(zipfolder)
    zipobject.extract_files()

    # Mosaicking & Clipping
    unzipfolder = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/UNZIP"
    vectorfile = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/BD_GABIR_2017_v3.shp"
    mos_clipObject = Mosaic_Clip(unzipfolder,vectorfile)
    mos_clipObject.parallelize()

    # Cloud status
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/MOS_CLIP"
    cld_Object = Cloud_Status(inPath)

    # Concatenate Images
    concatenateObject = Concatenate(inPath)
    concatenateObject.do()

    # GapFilling 
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF"
    GapFillingObject = GapFilling(inPath)
    GapFillingObject.do()

    # ========
    # DORDOGNE
    # ========

    # Unzipping
    zipfolder = "/media/je/SATA_1/Lab1/DORDOGNE/IMAGES"
    zipobject = ExtractZip(zipfolder)
    zipobject.extract_files()

    # Mosaic and Clipping
    unzipfolder = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/UNZIP"
    vectorfile = "/media/je/SATA_1/Lab1/DORDOGNE/SOURCE_VECTOR/ZONE_ENV.shp"
    mos_clipObject = Mosaic_Clip(unzipfolder,vectorfile)
    mos_clipObject.parallelize()

    # Cloud Status
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/MOS_CLIP"
    cld_Object = Cloud_Status(inPath)

    # Concatenate Images
    concatenateObject = Concatenate(inPath)
    concatenateObject.do()

    # GapFilling
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/GAPF"
    GapFillingObject = GapFilling(inPath)
    GapFillingObject.do()

    



    
    