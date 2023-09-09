import matplotlib
import sklearn 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import numpy as np 
import pandas as pd 

print("matplotlib version: {}".format(matplotlib.__version__))
print("sklearn version: {}".format(sklearn.__version__))
print("numpy version: {}".format(np.__version__))
print("pandas version: {}".format(pd.__version__))

# Read dataset 

df = pd.read_csv('train.csv')
area = df['1stFlrSF'] 
price = df['SalePrice']

# Subplot Mosaic example 1 
layout = """A.;BC"""
axs = plt.figure(layout="constrained", figsize=(10,8)).subplot_mosaic(
    mosaic=layout, height_ratios=[1,3.5], width_ratios=[3.5,1])
axs['A'].hist(area, orientation="vertical", color="grey", ec="grey", bins=250)
axs['C'].hist(price, orientation="horizontal", color="grey", ec="grey", bins=250)
axs['B'].scatter(area,price, color="red", s=1.5)

axs['A'].axis("off")
axs['C'].axis("off")

axs['B'].spines["top"].set_visible(False)
axs['B'].spines["right"].set_visible(False)
axs['B'].get_xaxis().tick_bottom()
axs['B'].get_yaxis().tick_left()
axs['B'].spines["left"].set_position(("outward", 10))
axs['B'].spines["bottom"].set_position(("outward", 10))
axs['B'].set_xlabel("AREA")
axs['B'].set_ylabel("SALE PRICE")

plt.suptitle("Relationship between area and sale price variable", fontsize=12)
plt.savefig("subplot_mosaic_example1.png")
plt.show()