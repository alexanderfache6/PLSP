import pyreadr

# result = pyreadr.read_r('/projectnb/planet/PLSP/data_paper/data/comp_hls/hls_3by3/50PCGI_2017_US-SRM__Santa_Rita_Mesquite.rda')
result = pyreadr.read_r('/projectnb/modislc/users/fache/data/planet/chunks/Walnut_Gulch_Kendall_Grasslands/temp/001/250101.rda')


print(len(list(result.values())))
dfs = list(result.values())

for df in dfs:
    print(df.columns)
    print(df.shape)
    print(df.head(10))