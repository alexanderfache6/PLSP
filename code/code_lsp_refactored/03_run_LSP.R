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

library(doMC)
library(doParallel)


########################################
args <- commandArgs()
print(args)

siteNumber <- as.numeric(args[4]) # site
chunkNumber <- as.numeric(args[5]) # chunk number; 1-200
# siteNumber <- 103 # NOTE temp when running in RStudio
# chunkNumber <- 50 # NOTE temp when running in RStudio
print(paste("siteNumber:", siteNumber))
print(paste("chunkNumber:", chunkNumber))


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

siteChunksDir <- paste0(params$setup$chunksDir, strSite)
print(paste("siteChunksDir:", siteChunksDir))

## Load chunk image
chunkNumberStringified <- sprintf("%03d", chunkNumber)
currentChunkFile <- list.files(path = siteChunksDir, pattern = glob2rx(paste0("*", chunkNumberStringified, ".rda")), full.names = TRUE)
load(currentChunkFile, verbose = TRUE) # this loads band1, band2, band3, band4, dates from .rda


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


##########################################
# Estimate phenometrics
numberOfPixelsPerBand <- dim(band1)[1]
phenoYears <- params$setup$phenStartYr:params$setup$phenEndYr
print(paste("phenoYears:", paste(phenoYears, collapse = ", ")))


phenoResultsMatrix <- matrix(NA, numberOfPixelsPerBand, 24 * length(phenoYears)) # each pixel gets 24 output metrics per year
print(paste("dim(phenoResultsMatrix):", paste(dim(phenoResultsMatrix), collapse = " x ")))

# NOTE for loaded site and chunk (represents single physical patch), calculate lsp per pixel across all days
for (i in 1:numberOfPixelsPerBand) {
  phenoResultsMatrix[i, ] <- DoPhenologyPlanet(band1[i, ], band2[i, ], band3[i, ], band4[i, ], datesAll, phenoYears, params) #, waterMask[i])
  if (i %% 10000 == 0) {
    print(paste("pixel:", i))
  }
}


# Save outputs
phenologyChunkDir <- paste0(params$setup$phenologyDir, strSite)
if (!dir.exists(phenologyChunkDir)) {
  dir.create(phenologyChunkDir, recursive = TRUE) # creates /planet/phenology/site/
  print(paste("created:", phenologyChunkDir))
} else {
  print(paste("exists:", phenologyChunkDir))
}

save(phenoResultsMatrix, file = paste0(phenologyChunkDir, "/chunk_phenology_", chunkNumberStringified, ".rda")) # creates /planet/phenology/site/chunk_phenology_001.rda
print(paste("saved:", paste0(phenologyChunkDir, "/chunk_phenology_", chunkNumberStringified, ".rda")))
