def main(): 
  missing_data()

def missing_data(): 
  # missing data
  print("Identifying missing values")

# example
import pandas as pd
import numpy as np
df = pd.DataFrame [[1.0, 5.0,10.0]
      , [2.0, 6.0, 11.0]
      , [3.0, np.nan, 12.0]
      , [4.0, 8.0, np.nan]
     ]
df.columns = ['A', 'B', 'C', 'D']
print(df)


if __name__ =="__main__":
  main()
