#!/bin/bash

# Reunion

python2.7 /media/je/SATA_1/Lab1/SCRIPTS/theia_download/theia_download.py -a \
/media/je/SATA_1/Lab1/SCRIPTS/theia_download/config_theia.cfg -w /media/je/SATA_1/Lab1/REUNION/IMAGES -c SENTINEL2 \
--lonmin 55.206631 --lonmax  55.846992 --latmin -21.398893 --latmax -20.862772 -d 2017-01-01 \
-f 2017-12-31

   

# Dordogne 

python2.7 /media/je/SATA_1/Lab1/SCRIPTS/theia_download/theia_download.py -a \
/media/je/SATA_1/Lab1/SCRIPTS/theia_download/config_theia.cfg -w /media/je/SATA_1/Lab1/DORDOGNE/IMAGES -c SENTINEL2 \
--lonmin 0.280860744075 --lonmax 0.966247054373 --latmin 44.6867526418 --latmax 45.1869602434 -d 2016-01-01 \
-f 2016-12-31