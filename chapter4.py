def main(): 
  missing_data()

def missing_data(): 
  # missing data
  print("Identifying missing values")

# example
import pandas as pd
df = [['A', 1.0, 5.0,10.0]
      , ['B', 2.0, 6.0, 11.0]
      , ['C', 3,0,,12.0]
      , ['D', 4.0, 8.0, ]
     ]
print(df)


if __name__ =="__main__":
  main()
