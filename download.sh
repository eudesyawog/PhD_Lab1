#!/bin/bash

# Reunion

python2.7 /media/je/SATA_1/Lab1/SCRIPTS/theia_download/theia_download.py -a \
/media/je/SATA_1/Lab1/SCRIPTS/theia_download/config_theia.cfg -w /media/je/SATA_1/Lab1/REUNION/IMAGES -c SENTINEL2 \
--lonmin 55.2123734002 --lonmax 55.8313833104 --latmin -20.8710970886 --latmax -21.3822618638 -d 2017-01-01 \
-f 2017-12-31

# Dordogne 

python2.7 /media/je/SATA_1/Lab1/SCRIPTS/theia_download/theia_download.py -a \
/media/je/SATA_1/Lab1/SCRIPTS/theia_download/config_theia.cfg -w /media/je/SATA_1/Lab1/DORDOGNE/IMAGES -c SENTINEL2 \
--lonmin 0.280860744075 --lonmax 0.966247054373 --latmin 45.1869602434 --latmax 44.6867526418 -d 2016-01-01 \
-f 2016-12-31