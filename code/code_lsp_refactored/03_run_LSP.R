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
# qsub -V -l h_rt=12:00:00 run_03.sh siteNumber chunk
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

library(sp)
library(raster)
library(terra)
library(sf)

library(rjson)
library(geojsonR)

library(doMC)
library(doParallel)


########################################
args <- commandArgs()
print(args)

siteNumber <- as.numeric(substr(args[3], 1, 3)) # site
currentChunk <- as.numeric(substr(args[3], 4, 6)) # chunk number; 1-200
# siteNumber <- 5; currentChunk <- 50


########################################
## Load parameters
params <- fromJSON(file = "/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/PLSP_Parameters_refactored.json")
source(params$setup$rFunctions)


########################################
## Get site name, image directory and coordinate
geojsonDir <- params$setup$geojsonDir

siteInfo <- GetSiteInfo(siteNumber, geojsonDir, params)
strSite <- siteInfo[[2]] # site name


siteChunksDir <- paste0(params$setup$chunksDir, strSite, "/chunk")
print(siteChunksDir)

## Load chunk image
chunkNumberStringified <- sprintf("%03d", currentChunk)
currentChunkFile <- list.files(path = siteChunksDir, pattern = glob2rx(paste0("*", chunkNumberStringified, ".rda")), full.names = TRUE)
load(currentChunkFile) # this loads band1, band2, band3, band4, dates from .rda


## Load water mask
waterRater <- raster(paste0(params$setup$outDir, strSite, "/water_mask_30_1.tif")) # TODO

numberOfChunks <- params$setup$numChunks
chunkSize <- length(waterRater) %/% numberOfChunks
if (currentChunk == numberOfChunks) {
  chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):length(waterRater))
} else {
  chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):(chunkSize * currentChunk))
}
waterMask <- values(waterRater)[chunkPixelIndices]


##########################################
# Estimate phenometrics
numberOfBandPixels <- dim(band1)[1]
phenoYears <- params$setup$phenStartYr:params$setup$phenEndYr

phenoResultsMatrix <- matrix(NA, numberOfBandPixels, 24 * length(phenoYears)) # each pixel gets 24 output metrics per year

# NOTE for loaded site and chunk, calculate lsp per pixel
for (i in 1:numberOfBandPixels) {
  phenoResultsMatrix[i, ] <- DoPhenologyPlanet(band1[i, ], band2[i, ], band3[i, ], band4[i, ], dates, phenoYears, params, waterMask[i])
  if (i %% 10000 == 0) {
    print(paste("pixel:", i))
  }
}


# Save outputs
phenologyChunkDir <- paste0(params$setup$mosaicsDir, strSite, "/chunk_phe")
if (!dir.exists(phenologyChunkDir)) {
  dir.create(phenologyChunkDir)
  print(paste("created:", phenologyChunkDir))
}

save(phenoResultsMatrix, file = paste0(phenologyChunkDir, "/chunk_phe_", chunkNumberStringified, ".rda"))
print(paste("saved:", paste0(phenologyChunkDir, "/chunk_phe_", chunkNumberStringified, ".rda")))
