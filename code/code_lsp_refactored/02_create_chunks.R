# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# A High Spatial Resolution Land Surface Phenology Dataset for AmeriFlux and NEON Sites
#
# 02: A script for PlanetScope image process; save mosaiced images into chunks
#
# Author: Minkyu Moon; moon.minkyu@gmail.com
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Submitted to the job for each site with shell script:
# #!/bin/bash
# echo Submitting $1
# R --vanilla < ~/02_make_chunks.R $1
#
# example submission command using default parameters:
# qsub -V -pe omp 28 -l h_rt=12:00:00 run_02.sh siteNumber
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

library(sp)
library(raster)
library(terra)
library(sf)

library(rjson)
library(geojsonR)

# library(doMC)
library(doParallel)

########################################
args <- commandArgs()
print(args)

siteNumber <- as.numeric(args[4])
# siteNumber <- 103 # NOTE temp when running in RStudio

########################################
## Load parameters
params <- fromJSON(file = "/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/PLSP_Parameters_refactored.json")
source(params$setup$rFunctions)


########################################
## Load mosaiced images
geojsonDir <- params$setup$geojsonDir

siteInfo <- GetSiteInfo(siteNumber, geojsonDir, params)
strSite <- siteInfo[[2]] # site name

siteMosaicDir <- paste0(params$setup$outDir, strSite, "/mosaic")
print(siteMosaicDir)

########################################
fileNamesMosaic <- list.files(path = siteMosaicDir, pattern = glob2rx("*mosaic.tif"))
filePathsMosaic <- list.files(path = siteMosaicDir, pattern = glob2rx("*mosaic.tif"), full.names = TRUE)

# Get dates
yy <- substr(fileNamesMosaic, 3, 4)
mm <- substr(fileNamesMosaic, 5, 6)
dd <- substr(fileNamesMosaic, 7, 8)
datesAll <- as.Date(paste(mm, dd, yy, sep = "/"), "%m/%d/%y")

print(length(datesAll))


# Divide mosaiced images into chunks
imgBase <- raster(paste0(params$setup$outDir, strSite, "/base_image.tif"))

numberOfChunks <- params$setup$numChunks # 200

chunkSize <- length(imgBase) %/% numberOfChunks
print(paste("numberOfChunks:", numberOfChunks, ", chunkSize:", chunkSize))

# Output directory
chunkDir <- paste0(params$setup$chunkDir, strSite)
if (!dir.exists(chunkDir)) {
  dir.create(chunkDir, recursive = TRUE)
  print(paste("created:", chunkDir))
}

# Directory for temporal outputs (which will be deleted at the end of the process)
chunkDirTemp <- paste0(params$setup$chunkDi, strSite, "/temp")
if (!dir.exists(chunkDirTemp)) {
  dir.create(chunkDirTemp, recursive = TRUE)
  print(paste("created:", chunkDirTemp))
}


########################################
# save chunks as temporal files

cluster <- makeCluster(params$setup$numCores)
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(raster)
  library(sp)
})

clusterExport(cluster, varlist = c("numberOfChunks", "chunkSize", "chunkDir", "chunkDirTemp"))


# NOTE iterate through all dates, get 4 bands
foreach(i = 1:length(datesAll)) %dopar% {
  # get bands for current date
  band1 <- values(raster::raster(filePathsMosaic[i], 1))
  band2 <- values(raster::raster(filePathsMosaic[i], 2))
  band3 <- values(raster::raster(filePathsMosaic[i], 3))
  band4 <- values(raster::raster(filePathsMosaic[i], 4))

  # NOTE slice pixels into 200 chunks, save date/chunk number to rda
  # output is small chunk files containing 4 bands for a date
  # create chunks for current date
  foreach(currentChunk = 1:numberOfChunks) %dopar% {
  # foreach(currentChunk = 1:numberOfChunks) %do% {
    currentChunkStringified <- sprintf("%03d", currentChunk)
    dirTemp <- paste0(chunkDirTemp, '/', currentChunkStringified)
    if (!dir.exists(dirTemp)) {
      dir.create(dirTemp)
    }

    if (currentChunk == numberOfChunks) {
      chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):length(imgBase)) # at last chunk, include remainder of pixels
    } else {
      chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):(chunkSize * currentChunk))
    }
    b1 <- band1[chunkPixelIndices]
    b2 <- band2[chunkPixelIndices]
    b3 <- band3[chunkPixelIndices]
    b4 <- band4[chunkPixelIndices]

    save(b1, b2, b3, b4, file = paste0(dirTemp, "/", yy[i], mm[i], dd[i], ".rda"))
    print(paste("saved:", paste0(dirTemp, "/", yy[i], mm[i], dd[i], ".rda")))
  }
  print(paste("done with date:", datesAll[i]))
}


########################################
# Load files for each chunk, merge then, and save
# load all dates for one chunk at a time and save as single rda


foreach(currentChunk = 1:numberOfChunks) %dopar% {
  currentChunkStringified <- sprintf("%03d", currentChunk)
  dirTemp <- paste0(chunkDirTemp, '/', currentChunkStringified)
  currentChunkAllDateFiles <- list.files(dirTemp, full.names = TRUE)

  if (currentChunk == numberOfChunks) {
    chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):length(imgBase))
  } else {
    chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):(chunkSize * currentChunk))
  }
  
  # check that the same number of chunks as dates were created
  if (length(currentChunkAllDateFiles) == length(datesAll)) {
    band1 <- matrix(NA, length(chunkPixelIndices), length(datesAll)) # rows are number of pixels, columnds are dates
    band2 <- matrix(NA, length(chunkPixelIndices), length(datesAll))
    band3 <- matrix(NA, length(chunkPixelIndices), length(datesAll))
    band4 <- matrix(NA, length(chunkPixelIndices), length(datesAll))
    for (i in 1:length(datesAll)) { # load each date and assign to chunk column
      load(currentChunkAllDateFiles[i])

      band1[, i] <- b1
      band2[, i] <- b2
      band3[, i] <- b3
      band4[, i] <- b4
    }
    # Save chunk - subset of pixels across all dates
    save(band1, band2, band3, band4, datesAll, file = paste0(chunkDir, "/chunk_", currentChunkStringified, ".rda"))
    print(paste("saved:", paste0(chunkDir, "/chunk_", currentChunkStringified, ".rda")))
  } else {
    print(paste("---------- failed on chunk number:", currentChunk, "----------"))
  }
}


########################################
## Remove temporary files
# system(paste0("rm -r ", chunkDirTemp))
