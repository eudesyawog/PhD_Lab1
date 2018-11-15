#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:06:54 2018

@author: je
"""
import zipfile
import os
import re
import subprocess
import fiona
import rasterio
import numpy as np
from affine import Affine
from itertools import product
import time
import concurrent.futures

class ExtractZip :
    """
    Extract All Images Zip Files
    zipfolder : Folder containing all zip files
    unzipfolder : Folder receiving unzip elements
    """
    def __init__(self, zipfolder, unzipfolder) :
        self._zipfolder = zipfolder
        self._unzipfolder = unzipfolder
        
    def extract_files(self) :  
        
        lstZip = [os.path.join(self._zipfolder,zipFile) for zipFile in os.listdir(self._zipfolder) if zipFile.endswith('.zip')]
        
        for zipFile in lstZip :
            zip_ref = zipfile.ZipFile(zipFile, 'r')
            zip_ref.extractall(self._unzipfolder)
            zip_ref.close()
    
class Clip_Organize :
    """
    Browse unzip folders, Clip bands of interest and corresponding clouds mask using 
    a vector file extent and Organize images by date
    """
    
    def __init__(self, folder, vectorfile, lstBand=["B2","B3","B4","B8","MASK"], product="FRE"):
        self._folder = folder
        self._vectorfile = vectorfile
        self._lstBand = lstBand
        self._product = product
        
    def execute(self):
        
        ds = fiona.open(self._vectorfile)
        ulx = ds.bounds[0]
        uly = ds.bounds[3]
        lrx = ds.bounds[2]
        lry = ds.bounds[1]
        ds = None

        compteur = 0

        outPath = os.path.dirname(os.path.dirname(self._folder))+os.sep+"CLIP"
        if not os.path.isdir(outPath):
            os.makedirs(outPath)
        # fichier = open(outPath+os.sep+"clouds.txt", "a")
        # fichier.write("Date \t Cloud Percentage \n")

        date = os.path.basename(self._folder).split('_')[1][:8]
        hour = os.path.basename(self._folder).split('_')[1][9:15]
        sensor = os.path.basename(self._folder).split('_')[0][-2:]
        outName = "{}_"+date+"_"+hour+"_S"+sensor+".tif"
        maskName = "CLM_"+date+"_"+hour+"_S"+sensor+".tif"
        # maskName2 = "CLMms_"+date+"_"+hour+"_S"+sensor+".tif"
            
        outFolder = outPath+os.sep+date
        if not os.path.isdir(outFolder):
            os.makedirs(outFolder)

        lstFiles = [File for File in os.listdir(self._folder) if File.endswith('.tif') and not File.endswith('B8A.tif') and File.split('_')[-2]==self._product]
        for band in self._lstBand:
            if band == "MASK" :
                for File in os.listdir(os.path.join(self._folder,"MASKS")):
                    if re.match("(.*)_{}".format("CLM_R1.tif"),File) and File.endswith(".tif"):
                        with rasterio.open(os.path.join(self._folder,"MASKS",File)) as ds:
                            if compteur == 0 :
                                originX = ds.affine[2]
                                originY = ds.affine[5]
                                pixelWidth = ds.affine[0]
                                pixelHeight = ds.affine[4]

                                xOffset = int((ulx- originX) / pixelWidth)
                                yOffset = int((uly - originY) / pixelHeight)
                                xsize = int((lrx - ulx)/pixelWidth)
                                ysize = int((uly - lry)/pixelWidth)

                            mask_array = ds.read(1,window=((yOffset,yOffset+ysize),(xOffset, xOffset+xsize)))
                            
                            # bin_array = np.empty(mask_array.shape,dtype=int)
                            # for m,n in product(range(mask_array.shape[0]),range(mask_array.shape[1])):
                            #     if bin(mask_array[m,n])[-1]=='1':
                            #         bin_array[m,n] = 1
                            #     else : 
                            #         bin_array[m,n] = 0
                            
                            mask_array = np.where(mask_array>0,1,0)

                            profile = ds.profile
                            profile.update(width=xsize,height=ysize,transform=(ulx,pixelWidth,0.,uly,0.,pixelHeight))
                        
                        with rasterio.open(os.path.join(outFolder,maskName),'w', **profile) as outds :
                            outds.write(mask_array.astype(rasterio.uint8),1)
                        
                        # with rasterio.open(os.path.join(outFolder,maskName2),'w', **profile) as outds :
                        #     outds.write(bin_array.astype(rasterio.uint8),1)
                        
                        # cld_percent = round((np.count_nonzero(mask_array==1) * 100) / (xsize * ysize))
                        # fichier.write("%s \t %s \t %s \n"%("S"+sensor,date,cld_percent))
                
            else :
                for File in lstFiles :
                    if re.match("(.*)_{}(.*)".format(band),File):
                        with rasterio.open(os.path.join(self._folder,File)) as ds:
                            if compteur == 0 :
                                originX = ds.affine[2]
                                originY = ds.affine[5]
                                pixelWidth = ds.affine[0]
                                pixelHeight = ds.affine[4]

                                xOffset = int((ulx - originX) / pixelWidth)
                                yOffset = int((uly - originY) / pixelHeight)
                                xsize = int((lrx - ulx)/pixelWidth)
                                ysize = int((uly - lry)/pixelWidth)

                            band_array = ds.read(1,window=((yOffset,yOffset+ysize),(xOffset, xOffset+xsize)))
                            band_array = np.where(band_array!=-10000,band_array+1,-10000)

                            profile = ds.profile
                            profile.update(width=xsize,height=ysize,nodata=-10000,transform=(ulx,pixelWidth,0.,uly,0.,pixelHeight))

                        with rasterio.open(os.path.join(outFolder,outName.format(band)),'w', **profile) as outds :
                            outds.write(band_array.astype(rasterio.int16),1)

            # command = "otbcli_ExtractROI -in %s -mode fit -mode.fit.vect %s -out %s int16"%(os.path.join(folder,file),self._vectorfile, os.path.join(outFolder,outName.format(band))) 
            # subprocess.call(command,shell=True)
            compteur += 1

        # fichier.close()

class Concatenate :
    """
    Concatenate images of the times series per band
    """

    def __init__(self, inPath, sitname, bands=["B2","B3","B4","B8","CLM"]):
        self._inPath = inPath
        self._site = sitname
        self._bands = bands
    
    def proceed (self):
        
        dates = [folder for folder in os.listdir(self._inPath) if os.path.isdir(os.path.join(self._inPath,folder))]
        dates.sort()

        outPath = os.path.dirname(self._inPath)+os.sep+"GAPF"
        if not os.path.isdir(outPath):
            os.makedirs(outPath)

        outName = "{}_{}_CONCAT.tif"

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
            Command += ['-out',os.path.join(outPath,outName.format(band,self._site))]
            Command += ['-ram','32768']
        
            subprocess.call(Command,shell=False)

class GapFilling :
    """
    Apply temporel gapfilling of Orfeo ToolBox on each band time series
    """
    def __init__(self, inPath, bands=["B2","B3","B4","B8"]):
        self._inPath = inPath
        self._bands = bands
    
    def proceed(self):

        lstFiles = [File for File in os.listdir(self._inPath) if File.startswith('B') and File.endswith('.tif')]
        lstFiles.sort()
        ptrn = lstFiles[0].split('_',1)[1]
        
        inName = os.path.join(self._inPath,"{}_"+ptrn)
        outName = os.path.join(self._inPath,"{}_"+ptrn.strip('.tif')+"_GAPF.tif")
        cloudfile = os.path.join(self._inPath,"CLM_"+ptrn)
        datesfile  = os.path.join(self._inPath,"dates.txt")

        for band in self._bands:
            Command = ['otbcli_ImageTimeSeriesGapFilling', '-in', inName.format(band),'-mask', cloudfile, 
                       '-out', outName.format(band), 'int16', '-comp', '1', '-it', 'linear','-id', datesfile, '-ram', '32768']
            
            subprocess.call(Command,shell=False)





if __name__ == '__main__':
    
    zipfolder = "/media/je/SATA_1/Lab1/REUNION/IMAGES"
    unzipfolder = "/media/je/SATA_1/Lab1/REUNION/IMAGES/UNZIP"
    
#    zipobject = ExtractZip(zipfolder,unzipfolder)
#    zipobject.extract_files()
    
    vectorfile = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/BD_GABIR_2017_v3.shp"

    # lstFolders = [os.path.join(unzipfolder,folder) for folder in os.listdir(unzipfolder) if os.path.isdir(os.path.join(unzipfolder,folder))]
    # lstFolders.sort()

    # NUM_WORKERS = 10
    # def work(folder):
    #     """
    #     Thread Function
    #     """
    #     clipObject = Clip_Organize(folder,vectorfile)
    #     clipObject.execute()
    #     return folder

    # start_time = time.time()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     futures = {executor.submit(work, folder) for folder in lstFolders}
    #     concurrent.futures.wait(futures)
    # end_time = time.time()

    # print("Time for proccess : %ssecs" % (end_time - start_time))

    # inPath = "/media/je/SATA_1/Lab1/REUNION/IMAGES/CLIP"
    # concatenateObject = Concatenate(inPath,sitname = "REUNION")
    # concatenateObject.proceed()

    inPath = "/media/je/SATA_1/Lab1/REUNION/IMAGES/GAPF"
    GapFillingObject = GapFilling(inPath)
    GapFillingObject.proceed()

    
    