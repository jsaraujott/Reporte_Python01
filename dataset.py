import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(
    iris.data,
    columns = iris.feature_names
)
df_iris["type"] = iris.target
df_iris.columns = [x.replace(" (cm)","").replace(" ","_") for x in df_iris.columns]

df_iris.to_csv("iris.csv", index = False)