import pandas as pd

df = pd.read_csv('/localdisk2/approx-topk/Data/UsedCars/val.csv')
df = df.head(200000)

df.to_csv('/localdisk2/approx-topk/Data/UsedCars/subset.csv', index=False)