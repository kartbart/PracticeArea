#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#%%
df = pd.read_csv("Wine.csv")
df

# %%
df.columns

# %%
df.corr()

# %%
plt.scatter(df["density"],df["residual sugar"])

# %%
sns.scatterplot(x="density", y="residual sugar", data=df)
plt.show()
# %%
df.describe()
df.shape

# %%
df.dtypes
# %%
sns.boxplot(df["density"])

# %%
sns.histplot(df["chlorides"])
# %%
fig = px.scatter(df, x="residual sugar", y="density")
fig.show()
# %%
