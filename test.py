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
fig = px.box(df, x="residual sugar")
fig.show()
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

X = df['residual sugar'].array.reshape(-1,1)  # Features
y = df['density']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Print the evaluation metrics
print('Mean Absolute Error:', mae)
print('R-squared:', r2)


# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', linewidth=2, label='Line of Best Fit')
plt.show()
