import glob
import os
import re
import gdal
import ogr
import osr
import rasterio 
from pprint import pprint
from datetime import datetime
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import subprocess

class Classifier :
    def __init__(self,inPath,ground_truth,niter=3,DateTime=None):
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
    
    def map(self,gridsize=(10000,10000),epsg=32740) :

        if not os.path.isdir(os.path.join(self._outPath,"MAP_%s"%self._datetime)):
            os.makedirs(os.path.join(self._outPath,"MAP_%s"%self._datetime))

        lstSpectral = [os.path.join(self._inPath,"GAPF",File) for File in os.listdir(os.path.join(self._inPath,"GAPF")) if File.endswith("GAPF.tif")]
        lstSpectral.sort()
        lstSpectral.extend([os.path.join(self._inPath,"INDICES",File) for File in os.listdir(os.path.join(self._inPath,"INDICES")) if File.endswith("GAPF.tif")])

        # lstEMP99 = glob.glob(os.path.join(self._inPath,"EMP_99")+os.sep+"*EMP.tif")
        # lstEMP99.sort()

        # lstTotal99 = []
        # lstTotal99.extend(lstSpectral)
        # lstTotal99.extend(lstEMP99)

        with rasterio.open(lstSpectral[0]) as ds:
            self._profile = ds.profile
            self._originX = int(ds.bounds[0])
            self._ulx = self._originX +1
            self._lry = int(ds.bounds[1])+1
            self._lrx = int(ds.bounds[2])-1
            self._originY = int(ds.bounds[3])
            self._uly = self._originY -1
        
        self._proj = osr.SpatialReference()
        self._proj.ImportFromEPSG(epsg)

        self._gridSize = gridsize

        rect = []
        for x,y in product(range(self._ulx,self._lrx,self._gridSize[0]),range(self._lry,self._uly,self._gridSize[1])):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x, y)
            if (x+self._gridSize[0] > self._lrx) and (y+self._gridSize[1] > self._uly) :
                ring.AddPoint(self._lrx, y)
                ring.AddPoint(self._lrx, self._uly)
                ring.AddPoint(x, self._uly)
            elif x+self._gridSize[0] > self._lrx :
                ring.AddPoint(self._lrx, y)
                ring.AddPoint(self._lrx, y + self._gridSize[1])
                ring.AddPoint(x, y + self._gridSize[1])
            elif y+self._gridSize[1] > self._uly :
                ring.AddPoint(x + self._gridSize[0], y)
                ring.AddPoint(x + self._gridSize[0], self._uly)
                ring.AddPoint(x, self._uly)
            else : 
                ring.AddPoint(x + self._gridSize[0], y)
                ring.AddPoint(x + self._gridSize[0], y + self._gridSize[1])
                ring.AddPoint(x, y + self._gridSize[1])
                
            ring.AddPoint(x, y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            rect.append(poly)

        print ("Map will be produced in %s blocks"%len(rect))
        mem_drv = gdal.GetDriverByName('MEM')
        # Spectral Map production
        print ("Spectral Map production")
        spectral_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(self._niter)))

#        for poly in rect :
#            extent = poly.GetEnvelope()
#            poly_ulx = extent[0]
#            poly_lry = extent[2]
#            poly_lrx = extent[1]
#            poly_uly = extent[3]

#            poly_xsize = int((poly_lrx-poly_ulx) /10)
#            poly_ysize = int((poly_lry-poly_uly) /-10)
#            # xOffset = int((poly_ulx - self._originX) / 10)
#            # yOffset = int((poly_uly - self._originY) / -10)

#            self._profile.update(count=1,dtype=rasterio.uint8,nodata=0,width=poly_xsize,height=poly_ysize,
#                           transform=(10.,0.,poly_ulx,0.,-10.,poly_uly))

#            spectral_samples = None
#            for File in lstSpectral :
#                with rasterio.open(File) as ds :
#                    count = ds.count
#                dest = mem_drv.Create('', poly_xsize, poly_ysize, count, gdal.GDT_Float32)
#                dest.SetGeoTransform((poly_ulx,10.,0.,poly_uly,0.,-10.))
#                dest.SetProjection(self._proj.ExportToWkt())
#                gdal.Warp(dest, File, outputBounds=(poly_ulx, poly_lry, poly_lrx, poly_uly))
#                for p in range(1,count+1):
#                    if spectral_samples is None :
#                        spectral_samples = dest.GetRasterBand(p).ReadAsArray().reshape(poly_ysize*poly_xsize)
#                    else :
#                        spectral_samples = np.column_stack((spectral_samples,dest.GetRasterBand(p).ReadAsArray().reshape(poly_ysize*poly_xsize)))
#                dest = None
            # print (spectral_samples.shape)
            # profile = self._profile.copy()
            # profile.update(count = spectral_samples.shape[1],dtype=rasterio.float32)
            # spectral_samples2 = spectral_samples.copy()
            # spectral_samples2 = spectral_samples2.reshape((poly_ysize,poly_xsize,spectral_samples.shape[1]))
            # with rasterio.open(os.path.join(self._outPath,"MAP_%s"%self._datetime,'spectral_samples_grid%s.tif'%(rect.index(poly)+1)),'w',**profile) as outDS :
                # for o in range(1,count+1):
                    # outDS.write(spectral_samples2[:,:,o].astype(rasterio.float32),o)
#            spectral_predict = spectral_model.predict(spectral_samples)
#            spectral_map = spectral_predict.reshape((poly_ysize,poly_xsize))
#            with rasterio.open(os.path.join(self._outPath,"MAP_%s"%self._datetime,'spectral_map_grid%s.tif'%(rect.index(poly)+1)),'w',**self._profile) as outDS :
#                outDS.write(spectral_map.astype(rasterio.uint8),1)
#            spectral_samples = None
#            print ('Block %s mapped'%(rect.index(poly)+1))
        
        lstMos = glob.glob(os.path.join(self._outPath,"MAP_%s"%self._datetime)+os.sep+'spectral_map_grid*.tif')
        lstMos.sort()
        pprint (lstMos)
        command = ["gdalwarp","-overwrite","-srcnodata","0","-dstnodata","0","-ot","Byte"]
        command.extend(lstMos)
        command+=[os.path.join(self._outPath,"MAP_%s"%self._datetime,'spectral_map.tif')]
        subprocess.call(command,shell=False) 
        
        # EMP 99 + Spectral Map production
        # print ("EMP 99 + Spectral Map production")
        # total99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(self._niter)))    
        # for poly in rect :
            # extent = poly.GetEnvelope()
            # poly_ulx = extent[0]
            # poly_lry = extent[2]
            # poly_lrx = extent[1]
            # poly_uly = extent[3]

            # poly_xsize = int((poly_lrx-poly_ulx) /10)
            # poly_ysize = int((poly_lry-poly_uly) /-10)
            # xOffset = int((poly_ulx - self._originX) / 10)
            # yOffset = int((poly_uly - self._originY) / -10)

            # self._profile.update(count=1,dtype=rasterio.int16,nodata=0,width=poly_xsize,height=poly_ysize,
                           # transform=(10.,0.,poly_ulx,0.,-10.,poly_uly))
            # total99_samples = None
            # for File in lstTotal99 :
                # with rasterio.open(File) as ds :
                    # count = ds.count
                # dest = mem_drv.Create('', poly_xsize, poly_ysize, count, gdal.GDT_Float32)
                # dest.SetGeoTransform((poly_ulx,10.,0.,poly_uly,0.,-10.))
                # dest.SetProjection(self._proj.ExportToWkt())
                # gdal.Warp(dest, File, outputBounds=(poly_ulx, poly_lry, poly_lrx, poly_uly))
                # for p in range(1,count+1):
                    # if total99_samples is None :
                        # total99_samples = dest.GetRasterBand(p).ReadAsArray().reshape(poly_xsize*poly_ysize)
                    # else :
                        # total99_samples = np.column_stack((total99_samples,dest.GetRasterBand(p).ReadAsArray().reshape(poly_xsize*poly_ysize)))
                # dest = None

            # total99_predict = total99_model.predict(total99_samples)
            # total99_map = total99_predict.reshape((poly_xsize,poly_ysize))
            # with rasterio.open(os.path.join(self._outPath,"MAP_%s"%self._datetime,'emp99+spectral_map_grid%s.tif'%(rect.index(poly)+1)),'w',**self._profile) as outDS :
                # outDS.write(total99_map.astype(rasterio.int16),1)
            # total99_samples = None
            # print ('Block %s mapped'%(rect.index(poly)+1))

        # lstMos = glob.glob(os.path.join(self._outPath,"MAP_%s"%self._datetime)+os.sep+'emp99+spectral_map_grid*.tif')
        # lstMos.sort()
        # pprint (lstMos)
        # command = ["gdalwarp","-overwrite","-srcnodata","0","-dstnodata","0","-ot","Byte"]
        # command.extend(lstMos)
        # command+=[os.path.join(self._outPath,"MAP_%s"%self._datetime,'spectral_map%s.tif')]
        # subprocess.call(command,shell=False) 
        
        # driver = ogr.GetDriverByName('ESRI Shapefile')
        # spatialRef = osr.SpatialReference()
        # spatialRef.ImportFromEPSG(32740)  
        # for p in range(len(rect)) :
        #     outDs = driver.CreateDataSource(os.path.join(self._outPath,"MAP_%s"%self._datetime,"map_grid_%s.shp"%(p+1)))
        #     outLayer = outDs.CreateLayer("map_grid_%s"%(p+1), srs=spatialRef,geom_type=ogr.wkbPolygon)
        #     featureDefn = outLayer.GetLayerDefn()
        #     outFeature = ogr.Feature(featureDefn)
        #     outFeature.SetGeometry(rect[p])
        #     outLayer.CreateFeature(outFeature)
        #     outFeature = None
        #     outDs = None
        #     print (rect[p].ExportToWkt())

if __name__ == '__main__':

    # ========
    # REUNION
    # ========

    # Classification
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/REUNION_GT_SAMPLES.shp"
    CO = Classifier(inPath,ground_truth,niter=3,DateTime="0201_1101")
    CO.map()

    # ========
    # DORDOGNE
    # ========

    # Classification
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/DORDOGNE/SOURCE_VECTOR/DORDOGNE_GT_SAMPLES_BUF-10_NOROADS.shp"
    CO = Classifier(inPath,ground_truth,niter=1,DateTime="0201_1102")
    CO.map(epsg=2154)
