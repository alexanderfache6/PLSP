# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# A High Spatial Resolution Land Surface Phenology Dataset for AmeriFlux and NEON Sites
#
# 03: A script for estimating phenometrics
#
# Author: Minkyu Moon; moon.minkyu@gmail.com
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Submitted to the job for a single chunk for a site with shell script:
# #!/bin/bash
# echo Submitting $1
# R --vanilla < ~/03_LSP_script.R $1
#
# example submission command using default parameters:
# qsub -V -l h_rt=12:00:00 run_03.sh siteNumber chunkNumber
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

library(sp)
library(raster)
library(terra)
library(sf)

library(rjson)
library(geojsonR)

library(doParallel)


########################################
args <- commandArgs()
print(args)

siteNumber <- as.numeric(args[4]) # site
# siteNumber <- 103 # NOTE temp when running in RStudio
print(paste("siteNumber:", siteNumber))

########################################
## Load parameters
params <- fromJSON(file = "/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/PLSP_Parameters_refactored.json")
source(params$setup$rFunctions)

########################################
## Get site name, image directory and coordinate
geojsonDir <- params$setup$geojsonDir

siteInfo <- GetSiteInfo(siteNumber, geojsonDir, params)
strSite <- siteInfo[[2]] # site name
print(paste("strSite:", strSite))

## Load chunk directory
siteChunksDir <- paste0(params$setup$chunksDir, strSite)
print(paste("siteChunksDir:", siteChunksDir))

phenologyChunkDir <- paste0(params$setup$phenologyDir, strSite)
if (!dir.exists(phenologyChunkDir)) {
  dir.create(phenologyChunkDir, recursive = TRUE) # creates /planet/phenology/site/
  print(paste("created:", phenologyChunkDir))
} else {
  print(paste("exists:", phenologyChunkDir))
}


########################################
## Setup phenology years and chunk count
phenoYears <- params$setup$phenStartYr:params$setup$phenEndYr
numberOfChunks <- params$setup$numChunks
print(paste("phenoYears:", paste(phenoYears, collapse = ", ")))
print(paste("numberOfChunks:", numberOfChunks))


########################################
## Setup parallel cluster
cluster <- makeCluster(params$setup$numCores)
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(raster)
  library(sp)
  library(foreach)
  library(doParallel)
  library(rjson)
})

clusterExport(cluster, varlist = c(
  "strSite",
  "siteChunksDir",
  "phenologyChunkDir",
  "phenoYears",
  "numberOfChunks",
  "params"
))

clusterEvalQ(cluster, {
  source(params$setup$rFunctions)
})

foreach(currentChunk = 1:numberOfChunks) %dopar% {
  currentChunkStringified <- sprintf("%03d", currentChunk)
  currentChunkFile <- list.files(path = siteChunksDir, pattern = glob2rx(paste0("*", currentChunkStringified, ".rda")), full.names = TRUE)
  log <- try(load(currentChunkFile, verbose = TRUE), silent = TRUE) # this loads band1, band2, band3, band4, dates from .rda
  if (inherits(log, "try-error")) {
    print(paste("---------- failed  to load chunk", strSite, currentChunkStringified, "----------"))
    return(NULL)
  }

  numberOfPixelsPerBand <- dim(band1)[1]
  phenoResultsMatrix <- matrix(NA, numberOfPixelsPerBand, 24 * length(phenoYears)) # each pixel gets 24 output metrics per year
  print(paste("dim(phenoResultsMatrix):", paste(dim(phenoResultsMatrix), collapse = " x ")))


  # # NOTE arid regions assuming no water issues, manually checked in zoomed in maps

  # ## Load water mask
  # waterRater <- raster(paste0(params$setup$outDir, strSite, "/water_mask_30_1.tif")) # TODO

  # numberOfChunks <- params$setup$numChunks
  # chunkSize <- length(waterRater) %/% numberOfChunks
  # if (chunkNumber == numberOfChunks) {
  #   chunkPixelIndices <- c((chunkSize * (chunkNumber - 1) + 1):length(waterRater))
  # } else {
  #   chunkPixelIndices <- c((chunkSize * (chunkNumber - 1) + 1):(chunkSize * chunkNumber))
  # }
  # waterMask <- values(waterRater)[chunkPixelIndices]


  # NOTE for loaded site and chunk (represents single physical patch), calculate lsp per pixel across all days
  for (i in 1:numberOfPixelsPerBand) {
    phenoResultsMatrix[i, ] <- DoPhenologyPlanet(band1[i, ], band2[i, ], band3[i, ], band4[i, ], datesAll, phenoYears, params) # , waterMask[i])
    # if (i %% 10000 == 0) {
    #   print(paste("[", strSite, "]", "chunk:", currentChunkStringified, "pixel progress:", i, "/", numberOfPixelsPerBand))
    # }
  }

  outFile <- paste0(phenologyChunkDir, "/chunk_phenology_", currentChunkStringified, ".rda")
  save(phenoResultsMatrix, file = outFile) # creates /planet/phenology/site/chunk_phenology_001.rda
  print(paste("saved:", outFile))
} # end %dopar%

stopCluster(cluster)

numChunksSaved <- length(list.files(path = phenologyChunkDir, pattern = glob2rx("*.rda")))
print(paste("----------", "[", strSite, "]", "number of chunks saved:", numChunksSaved, "/", numberOfChunks, "----------"))
