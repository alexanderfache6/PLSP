import pyreadr

result = pyreadr.read_r('/projectnb/planet/PLSP/data_paper/data/comp_hls/hls_3by3/50PCGI_2017_US-SRM__Santa_Rita_Mesquite.rda')

print(len(list(result.values())))
df = list(result.values())[0]

print(df.columns)
print(df.shape)
print(df.head(100))