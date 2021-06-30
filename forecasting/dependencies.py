# -*- coding: utf-8 -*-
import os
from rpy2.robjects.packages import importr
from pathlib import Path, PureWindowsPath


def download_packages():
    base_path = PureWindowsPath(str(Path.home()), "anaconda3/Lib/R/library/")
    if not os.path.isdir(base_path):
        raise ('set path where anaconda packages are installed manually and remove this check')

    packages = ['forecast', 'forecastHybrid', 'stats']
    utils = importr('utils')
    utils.chooseCRANmirror(ind=91)
    for pkg in packages:
        path = PureWindowsPath(base_path, pkg)
        if not os.path.isdir(path):
            print('{} not found. downloading ...'.format(pkg))
            utils.install_packages(pkg)
        else:
            print('{} found. not downloading'.format(pkg))
