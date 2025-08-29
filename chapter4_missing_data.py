def main(): 
# missing data
print("Identifying missing values")

# df
import pandas as pd
import numpy as np
df = pd.DataFrame([[1.0, 5.0, 10.0]
      , [2.0, 6.0, 11.0]
      , [3.0, np.nan,12.0]
      , [4.0, 8.0, np.nan]
])

df.columns = ['A', 'B', 'C']
print(df, "\n")

# idenfying number of missing values per column
print("Number of missing values per attribute\n", df.isnull().sum(), "\n")

# eliminating NaN
print("Dropping rows with missing values\n", df.dropna(axis=0), "\n")
print("Dropping columns with missing values\n", df.dropna(axis=1))

# mean imputation
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values= np.nan, strategy= 'mean')
imr = imr.fit(df.values)
imp_df = imr.transform(df.values)
print("Mean imputation\n", imp_df, "\n")

if __name__ =="__main__":
  main()
