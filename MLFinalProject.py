'''
Arikuzzaman Idrisy
ML - FINAL PROJECT
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lockAndLoad():
    file = open('SDSS_DR14.csv','r')

    df = pd.read_csv(file,
                    header =0 ,
                    usecols=["u","g","r","i","z","redshift","class"]
                    )
    print(df)
    useCols = ["u","g","r","i","z","redshift","class"]


    dfret = df.reindex(columns= useCols)

    print(dfret)

    data = dfret.values #Numpy Array of the data

    return data


def main():
    print(lockAndLoad())

    
main()
