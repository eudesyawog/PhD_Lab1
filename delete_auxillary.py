import glob
import os
from pprint import pprint

class AuxXML :
    """
    List aux.xml files in inputs folder and delete them
    """

    def __init__(self, lstFolders=""):
        object.__init__(self)
        self.__lstFolders = lstFolders
    
    def __str__(self) : 
        return "Actual Folders : " + self.__lstFolders
    
    def __repr__(self):
        str(self.__dict__)

    def add_folder(self, inFolder):
        self.__lstFolders.append(inFolder)
        pprint (self.__lstFolders)
    
    def clean(self):
        lstAux = []
        for Folder in self.__lstFolders :
            lstAux.extend(glob.glob(Folder+os.sep+"**"+os.sep+"*aux.xml",recursive=True))
        print (str(len(lstAux)) + " Auxillary File(s) found and deleted")
        for File in lstAux :
            os.remove(File)

if __name__=="__main__" : 
    lstFolders = ["/home/je","/media/je/SATA_1","/media/je/SATA_2","/media/je/JE3"]
    AuxXMLObject = AuxXML(lstFolders)
    AuxXMLObject.clean()
