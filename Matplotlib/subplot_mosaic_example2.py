import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("matplotlib version: {}".format(matplotlib.__version__))
print("numpy version: {}".format(np.__version__))
print("pandas version: {}".format(pd.__version__))

# Read dataset
df = pd.read_csv("train.csv")

layout = [["hist_one", "main"], ["hist_two", "main"]]
build_type = df.groupby(["BldgType"], as_index=False).agg({"SalePrice": "mean"})
fig = plt.figure(layout="constrained", figsize=(15, 7))
axs = fig.subplot_mosaic(
    mosaic=layout,
    per_subplot_kw={
        ("hist_one", "hist_two"): {
            "xlim": (0, 11),
            "xticks": np.arange(0, 12, 2),
            "yticks": np.arange(0, 800000, 100000),
            "yticklabels": [
                "{:.0f}".format(y / 1000) + "K" for y in range(0, 800000, 100000)
            ],
            "xlabel": "Quality",
            "ylabel": "Sale Price",
        },
        "main": {"xlabel": "Building Type", "ylabel": "Sale Price"},
    },
)

axs["hist_one"].scatter(df.OverallCond, df.SalePrice, color="red")
axs["hist_two"].scatter(df.OverallQual, df.SalePrice, color="red")
axs["main"].bar(
    x=build_type.BldgType, height=build_type.SalePrice, color="red", alpha=0.5
)
plt.savefig("subplot_mosaic_example2.png")
plt.show()
