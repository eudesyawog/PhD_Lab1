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
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing, disk

# class Geodesic_Morpho_Operation :
#     """
#     Performs geoedesic morphological operations on a grayscale input image
#     """

#     def __init__(self, inPath, NUM_WORKERS=5):
#         """
#         """
#         self._inPath = inPath
#         self._outPath = os.path.dirname(inPath)+os.sep+"EMP_Geodesic_%s"%os.path.basename(inPath).split('_')[-1]
#         if not os.path.isdir(self._outPath):
#             os.makedirs(self._outPath)
#         self._NUM_WORKERS = NUM_WORKERS
    
#     def _process_mp (self,File):
        
#         tmp_folder = self._outPath+os.sep+os.path.basename(File).split('_',1)[0]+"_EMP"
#         tmp_file = os.path.basename(File).replace(".tif","_C%s_%s_%s.tif")
#         if not os.path.isdir(tmp_folder):
#             os.makedirs(tmp_folder)
        
#         with rasterio.open(File) as ds:
#             count = ds.count       
#             for i in range(1,count+1):
#                 for j in range(3,8,2):
#                     for k in ("dilate","erode","opening","closing"):
#                         selem = disk(j)
#                         eroded = erosion(ds.read(i), selem)

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
        # emp99_data = np.load(self._emp99_data)
        spectral_data = np.load(self._spectral_data)
        # total99_data = np.column_stack((emp99_data,spectral_data))

        gt = gpd.read_file(self._gt)
        dicClass = gt.groupby(["Code"])["Name"].unique().to_dict()
        lstClass = list(dicClass.keys())
        # dicCount = gt.groupby(["Code"])["Code"].count().to_dict()

        if not os.path.isdir(os.path.join(self._outPath,"MODELS_%s"%self._datetime)):
            os.makedirs(os.path.join(self._outPath,"MODELS_%s"%self._datetime))
        if not os.path.isdir(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime)):
            os.makedirs(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime))

        # outCSV = os.path.join(self._outPath,"results_summary_%s.csv"%self._datetime)
        # outCSViter = os.path.join(self._outPath,"results_iterations_%s.csv"%self._datetime)
        # outdic = {}
        # fichier_txt = os.path.join(self._outPath,"MODELS_%s"%self._datetime,"best_params_%s.txt"%self._datetime)
        
        for i in range(self._niter):
            if not os.path.isfile(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1))) and not os.path.isfile(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"validation_samples_iteration%s.shp"%(i+1))) and not os.path.isfile(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1))) :
                train_ID = []
                validation_ID = []
                test_ID = []
                for c in lstClass :
                    lstID = gt["ID"][gt["Code"] == c].tolist()
                    random.shuffle(lstID)
                    random.shuffle(lstID)
                    train_ID.extend(lstID[:round(len(lstID)*0.5)]) #Train Validation Test proportion 50% 20% 30%
                    validation_ID.extend(lstID[round(len(lstID)*0.5):round(len(lstID)*0.7)])
                    test_ID.extend(lstID[round(len(lstID)*0.7):])

                train_df = gt[gt["ID"].isin(train_ID)]
                train_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1)))
                
                validation_df = gt[gt["ID"].isin(validation_ID)]
                validation_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"validation_samples_iteration%s.shp"%(i+1)))
            
                test_df = gt[gt["ID"].isin(test_ID)]
                test_df.to_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1)))
        print ("%s Random Split OK !"%self._niter)
        
        
        for i in range(self._niter):
            print ("Iteration %s"%(i+1))

            train_df = gpd.read_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"train_samples_iteration%s.shp"%(i+1)))
            train_ID = train_df["ID"].tolist()

            validation_df = gpd.read_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"validation_samples_iteration%s.shp"%(i+1)))
            validation_ID = validation_df["ID"].tolist()

            test_df = gpd.read_file(os.path.join(self._outPath,"SAMPLES_%s"%self._datetime,"test_samples_iteration%s.shp"%(i+1)))
            test_ID = test_df["ID"].tolist()

            # Merge Train & Validation sets
            merged_ID = []
            merged_ID.extend(train_ID)
            merged_ID.extend(validation_ID)
            merged_ix = np.where(np.isin(self._gt_ID, merged_ID))
            merged_labels = self._gt_labels[merged_ix]

            train_ix = np.where(np.isin(self._gt_ID, train_ID))
            train_labels = self._gt_labels[train_ix]
            validation_ix = np.where(np.isin(self._gt_ID, validation_ID))
            validation_labels =  self._gt_labels[validation_ix]
        
            # Predefined split
            fold = []
            fold.extend([-1]*train_labels.size)
            fold.extend([0]*validation_labels.size)
            ps = PredefinedSplit(test_fold=fold)

            # GridSearch
            rf = RandomForestClassifier()
            n_estimators = [200,300,400,500]
            max_depth = [20,40,60,80,100,None]
            tuned_parameters = {'n_estimators': n_estimators,
                                'max_depth': max_depth}
            # Spectral Bands
            if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1))):
                spectral_merged_samples = spectral_data[merged_ix]
                grid = GridSearchCV(rf, param_grid=tuned_parameters, cv=ps, n_jobs=-1,verbose=3)
                grid.fit(spectral_merged_samples, merged_labels)
                joblib.dump(grid, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1)))
                spectral_merged_samples = None
                
                print("Best score: {}".format(grid.best_score_))
                print("Best set of hyperparameters: {} \n".format(grid.best_params_))
                # spectral_best_param = (grid.best_params_['n_estimators'],grid.best_params_['max_depth'])
                # fi = open(fichier_txt,'a')
                # fi.write("Iteration %s"%(i+1))
                # fi.write("\n Spectral bands : %s"%str(spectral_best_param))
                # fi.close()    
            else:
                spectral_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'spectral_model_iteration%s.pkl'%(i+1)))

            # EMP99
            # if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1))):
            #     emp99_merged_samples = emp99_data[merged_ix]
            #     grid = GridSearchCV(rf, param_grid=tuned_parameters, cv=ps, n_jobs=-1,pre_dispatch=12,verbose=3)
            #     grid.fit(emp99_merged_samples, merged_labels)
            #     joblib.dump(grid, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1)))
            #     emp99_merged_samples = None

            #     print("Best score: {}".format(grid.best_score_))
            #     print("Best set of hyperparameters: {}".format(grid.best_params_))
            #     emp99_best_param = (grid.best_params_['n_estimators'],grid.best_params_['max_depth'])
            #     fi = open(fichier_txt,'a')
            #     fi.write("\n EMP 99 : %s"%str(emp99_best_param))
            #     fi.close()

                # res = grid.cv_results_['mean_test_score'].reshape(len(n_estimators),len(max_depth))
                # X, Y = np.meshgrid(max_depth, n_estimators)
                # fig, ax = plt.subplots(1,1)
                # cp = ax.contourf(X, Y, res)
                # ax.scatter(grid.best_params_['max_depth'],grid.best_params_['n_estimators'],color='k')
                # ax.set_xscale("log")
                # ax.set_yscale("log")
                # ax.set_xlabel("max_depth")
                # ax.set_ylabel("n_estimators")
                # fig.colorbar(cp)
                # plt.title(grid.best_params_)
                # plt.savefig("test.png",dpi=150)
                # plt.show()
                
            # else:
            #     emp99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99_model_iteration%s.pkl'%(i+1)))
            
            # EMP99 + Spectral Bands
            # if not os.path.isfile(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1))):
            #     total99_merged_samples = total99_data[merged_ix]
            #     grid = GridSearchCV(rf, param_grid=tuned_parameters, cv=ps, n_jobs=-1, pre_dispatch=6,verbose=3)
            #     grid.fit(total99_merged_samples, merged_labels)
            #     joblib.dump(grid, os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1)))
            #     total99_merged_samples = None

            #     print("Best score: {}".format(grid.best_score_))
            #     print("Best set of hyperparameters: {}".format(grid.best_params_))
            #     total99_best_param = (grid.best_params_['n_estimators'],grid.best_params_['max_depth'])
            #     fi = open(fichier_txt,'a')
            #     fi.write("\n EMP 99 + Spectral bands : %s"%str(total99_best_param))
            #     fi.close()       
            # else:
            #     total99_model = joblib.load(os.path.join(self._outPath,"MODELS_%s"%self._datetime,'emp99+spectral_model_iteration%s.pkl'%(i+1)))
            
        #     # Testing
        #     test_ix = np.where(np.isin(self._gt_ID, test_ID))
        #     test_labels =  self._gt_labels[test_ix]

        #     # Predict & Metrics Spectral Bands
        #     spectral_test_samples = spectral_data[test_ix]    
        #     spectral_predict = spectral_model.predict(spectral_test_samples)
        #     outdic.setdefault('Input',[]).append("Spectral Bands")
        #     # outdic.setdefault('Best Parameters',[]).append(spectral_best_param)
        #     outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, spectral_predict))
        #     outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, spectral_predict))
        #     outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, spectral_predict,average='weighted'))
        #     spectral_per_class = f1_score(test_labels, spectral_predict,average=None)
        #     spectral_test_samples = None

        #     # Predict & Metrics EMP99
        #     emp99_test_samples = emp99_data[test_ix]
        #     emp99_predict = emp99_model.predict(emp99_test_samples)
        #     outdic.setdefault('Input',[]).append("EMP99")
        #     # outdic.setdefault('Best Parameters',[]).append(emp99_best_param)
        #     outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, emp99_predict))
        #     outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, emp99_predict))
        #     outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, emp99_predict,average='weighted'))
        #     emp99_per_class = f1_score(test_labels, emp99_predict,average=None)
        #     emp99_test_samples = None

        #     # Predict & Metrics EMP99 + Spectral Data
        #     total99_test_samples = total99_data[test_ix]
        #     total99_predict = total99_model.predict(total99_test_samples)
        #     outdic.setdefault('Input',[]).append("EMP99 + Spectral Bands")
        #     # outdic.setdefault('Best Parameters',[]).append(total99_best_param)
        #     outdic.setdefault('Overall Accuracy',[]).append(accuracy_score(test_labels, total99_predict))
        #     outdic.setdefault('Kappa Coefficient',[]).append(cohen_kappa_score(test_labels, total99_predict))
        #     outdic.setdefault('F-Measure',[]).append(f1_score(test_labels, total99_predict,average='weighted'))
        #     total99_per_class = f1_score(test_labels, total99_predict,average=None)
        #     total99_test_samples = None

        #     print ("EMP99 | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
        #         round(outdic['Overall Accuracy'][i*3],3),round(outdic['Kappa Coefficient'][i*3],3),round(outdic['F-Measure'][i*3],3)))
            
        #     print ("Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
        #         round(outdic['Overall Accuracy'][i*3+1],3),round(outdic['Kappa Coefficient'][i*3+1],3),round(outdic['F-Measure'][i*3+1],3)))

        #     print ("EMP99 + Spectral Bands | Overall accuracy: %s; Kappa Coefficient: %s; F-Measure: %s"%(
        #         round(outdic['Overall Accuracy'][i*3+2],3),round(outdic['Kappa Coefficient'][i*3+2],3),round(outdic['F-Measure'][i*3+2],3)))

        # outdf = pd.DataFrame.from_dict(outdic)
        # outdf.to_csv(outCSViter, index=False)

        # mean_df = outdf.groupby(["Input"])[["Overall Accuracy","Kappa Coefficient","F-Measure"]].mean()
        # std_df = outdf.groupby(["Input"])[["Overall Accuracy","Kappa Coefficient","F-Measure"]].std()

        # df1 = pd.DataFrame({'Input': ["EMP99", "Spectral Bands", "EMP99 + Spectral Bands",""],
        #             'Overall Accuracy': ["%s +/- %s"%(round(mean_df.loc['EMP99']['Overall Accuracy'],3),round(std_df.loc['EMP99']['Overall Accuracy'],3)),
        #                                  "%s +/- %s"%(round(mean_df.loc['Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['Spectral Bands']['Overall Accuracy'],3)),
        #                                  "%s +/- %s"%(round(mean_df.loc['EMP99 + Spectral Bands']['Overall Accuracy'],3),round(std_df.loc['EMP99 + Spectral Bands']['Overall Accuracy'],3)),
        #                                  ""],
                                         
        #             'Kappa Coefficient' : ["%s +/- %s"%(round(mean_df.loc['EMP99']['Kappa Coefficient'],3),round(std_df.loc['EMP99']['Kappa Coefficient'],3)),
        #                                    "%s +/- %s"%(round(mean_df.loc['Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['Spectral Bands']['Kappa Coefficient'],3)),
        #                                    "%s +/- %s"%(round(mean_df.loc['EMP99 + Spectral Bands']['Kappa Coefficient'],3),round(std_df.loc['EMP99 + Spectral Bands']['Kappa Coefficient'],3)),
        #                                    ""],

        #             'F-Measure' : ["%s +/- %s"%(round(mean_df.loc['EMP99']['F-Measure'],3),round(std_df.loc['EMP99']['F-Measure'],3)),
        #                            "%s +/- %s"%(round(mean_df.loc['Spectral Bands']['F-Measure'],3),round(std_df.loc['Spectral Bands']['F-Measure'],3)),
        #                            "%s +/- %s"%(round(mean_df.loc['EMP99 + Spectral Bands']['F-Measure'],3),round(std_df.loc['EMP99 + Spectral Bands']['F-Measure'],3)),
        #                            ""]})
        # df1.to_csv(outCSV, index=False)

        # dic2 = {}
        # dic2.setdefault('Per Class F-Measure',[]).append("EMP99")
        # for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
        #     dic2.setdefault('Class %s'%(v),[]).append(round(emp99_per_class[f],3))

        # dic2.setdefault('Per Class F-Measure',[]).append("Spectral Bands")
        # for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
        #     dic2.setdefault('Class %s'%(v),[]).append(round(spectral_per_class[f],3))
        
        # dic2.setdefault('Per Class F-Measure',[]).append("EMP99 + Spectral Bands")
        # for v,f in zip(list(dicClass.keys()),range(len(list(dicClass.keys())))):
        #     dic2.setdefault('Class %s'%(v),[]).append(round(total99_per_class[f],3))
        
        # dic2.setdefault('Per Class F-Measure',[]).append("")
        # for v in list(dicClass.keys()):
        #     dic2.setdefault('Class %s'%(v),[]).append("")
            
        # df2 = pd.DataFrame.from_dict(dic2)
        # df2.to_csv(outCSV, mode ='a', index=False)

        # df3 = pd.DataFrame({'Class': [str(v) for v in list(dicClass.keys())],
        #                     'Name' : [dicClass[v][0] for v in list(dicClass.keys())],
        #                     'Objects': [dicCount[v] for v in list(dicClass.keys())],
        #                     'Pixels': [np.count_nonzero(self._gt_labels==v) for v in list(dicClass.keys())]
        #                     })
        # df3.to_csv(outCSV, mode='a', index=False)

if __name__ == '__main__':

    # ========
    # REUNION
    # ========
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT/PCA_99"
    # morpho = Geodesic_Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    # Classification
    inPath = "/media/je/SATA_1/Lab1/REUNION/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/REUNION/BD_GABIR_2017_v3/REUNION_GT_SAMPLES.shp"
    # CO = Classifier(inPath,ground_truth)
    # CO.classify()

    # ========
    # DORDOGNE
    # ========
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT/PCA_99"
    # morpho = Geodesic_Morpho_Operation(inPath)
    # morpho.create_mp()
    # morpho.create_emp()

    # Classification
    inPath = "/media/je/SATA_1/Lab1/DORDOGNE/OUTPUT"
    ground_truth = "/media/je/SATA_1/Lab1/DORDOGNE/SOURCE_VECTOR/DORDOGNE_GT_SAMPLES_BUF-10_NOROADS.shp"
    CO = Classifier(inPath,ground_truth)
    CO.classify()
