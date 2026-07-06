# NOTE creates 24 layer/metric tif phenologyChunkFiles per site per year
# groups together phenology_chunk files into single raster

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# A High Spatial Resolution Land Surface Phenology Dataset for AmeriFlux and NEON Sites
#
# 04: A script for saving data layers into GeoTiff format
#
# Author: Minkyu Moon; moon.minkyu@gmail.com
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Submitted to the job for a site with shell script:
# #!/bin/bash
# echo Submitting $1
# R --vanilla < ~/04_generate.R $1
#
# example submission command using default parameters:
# qsub -V -pe omp 4 -l h_rt=12:00:00 run_04.sh siteNumber
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

library(sp)
library(raster)
library(terra)
library(sf)
library(rjson)

########################################
args <- commandArgs()
print(args)

siteNumber <- as.numeric(args[4])
# siteNumber <- 103 # NOTE temp when running in RStudio
print(paste("siteNumber:", siteNumber))

########################################
## Load parameters
params <- fromJSON(file = "/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/PLSP_Parameters_refactored.json")
source(params$setup$rFunctions)

productTable <- read.csv(params$setup$productTable, header = TRUE, stringsAsFactors = FALSE)
phenoYears <- params$setup$phenStartYr:params$setup$phenEndYr
print(paste("phenoYears:", paste(phenoYears, collapse = ", ")))

########################################
geojsonDir <- params$setup$geojsonDir

siteInfo <- GetSiteInfo(siteNumber, geojsonDir, params)
strSite <- siteInfo[[2]] # site name
print(paste("strSite:", strSite))

phenologyChunkDir <- paste0(params$setup$phenologyDir, strSite)
print(paste("phenologyChunkDir:", phenologyChunkDir))

phenologyChunkFiles <- list.files(path = phenologyChunkDir, pattern = glob2rx("*.rda"), full.names = TRUE)
print(paste("length(phenologyChunkFiles):", length(phenologyChunkFiles)))

########################################
# Get all images to process
imgBase <- raster(paste0(params$setup$mosaicsDir, strSite, "/base_image.tif"))
numberOfPixels <- length(imgBase)
numberOfChunks <- params$setup$numChunks
chunkSize <- numberOfPixels %/% numberOfChunks
print(paste("numberOfPixels:", numberOfPixels))
print(paste("numberOfChunks:", numberOfChunks))
print(paste("chunkSize:", chunkSize))

########################################
# Save
phenologyGeotiffDir <- paste0(params$setup$phenologyGeotiffDir, strSite) # creates /planet/phenologyGeotiff/site/
if (!dir.exists(phenologyGeotiffDir)) {
  dir.create(phenologyGeotiffDir)
  print(paste("created:", phenologyGeotiffDir))
} else {
  print(paste("exists:", phenologyGeotiffDir))
}

for (currentYearIdx in 1:length(phenoYears)) {
  print(paste("[", phenoYears[currentYearIdx], "]", "starting"))

  layer01 <- matrix(NA, numberOfPixels, 1)
  layer02 <- matrix(NA, numberOfPixels, 1)
  layer03 <- matrix(NA, numberOfPixels, 1)
  layer04 <- matrix(NA, numberOfPixels, 1)
  layer05 <- matrix(NA, numberOfPixels, 1)
  layer06 <- matrix(NA, numberOfPixels, 1)
  layer07 <- matrix(NA, numberOfPixels, 1)
  layer08 <- matrix(NA, numberOfPixels, 1)
  layer09 <- matrix(NA, numberOfPixels, 1)
  layer10 <- matrix(NA, numberOfPixels, 1)
  layer11 <- matrix(NA, numberOfPixels, 1)
  layer12 <- matrix(NA, numberOfPixels, 1)
  layer13 <- matrix(NA, numberOfPixels, 1)
  layer14 <- matrix(NA, numberOfPixels, 1)
  layer15 <- matrix(NA, numberOfPixels, 1)
  layer16 <- matrix(NA, numberOfPixels, 1)
  layer17 <- matrix(NA, numberOfPixels, 1)
  layer18 <- matrix(NA, numberOfPixels, 1)
  layer19 <- matrix(NA, numberOfPixels, 1)
  layer20 <- matrix(NA, numberOfPixels, 1)
  layer21 <- matrix(NA, numberOfPixels, 1)
  layer22 <- matrix(NA, numberOfPixels, 1)
  layer23 <- matrix(NA, numberOfPixels, 1)
  layer24 <- matrix(NA, numberOfPixels, 1)
  print(paste("[", phenoYears[currentYearIdx], "]", "layers created"))

  for (currentChunk in 1:numberOfChunks) {
    currentChunkFile <- paste0(phenologyChunkDir, "/chunk_phenology_", currentChunkStringified, ".rda")
    log <- try(load(currentChunkFile, verbose = TRUE), silent = FALSE) # loads phenoResultsMatrix from 03_run_LSP.R
    if (inherits(log, "try-error")) {
      print(paste("[", phenoYears[currentYearIdx], "]", "error loading chunk", currentChunkFile))
      next
    }

    if (currentChunk == numberOfChunks) {
      chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):length(imgBase)) # at last chunk, include remainder of pixels
    } else {
      chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):(chunkSize * currentChunk))
    }

    chunkStartIdx <- chunkPixelIndices[1]
    chunkEndIdx <- chunkPixelIndices[length(chunkPixelIndices)]

    # each row represents pixel indices for current chunk - assign columns in batches of 24 (24 metrics) per year
    layer01[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 1)]
    layer02[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 2)]
    layer03[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 3)]
    layer04[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 4)]
    layer05[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 5)]
    layer06[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 6)]
    layer07[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 7)]
    layer08[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 8)]
    layer09[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 9)]
    layer10[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 10)]
    layer11[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 11)]
    layer12[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 12)]
    layer13[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 13)]
    layer14[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 14)]
    layer15[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 15)]
    layer16[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 16)]
    layer17[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 17)]
    layer18[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 18)]
    layer19[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 19)]
    layer20[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 20)]
    layer21[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 21)]
    layer22[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 22)]
    layer23[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 23)]
    layer24[chunkStartIdx:chunkEndIdx, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 24)]
  }
  print(paste("[", phenoYears[currentYearIdx], "]", "chunks updated"))

  # update raster values
  raster01 <- raster::setValues(imgBase, layer01)
  raster02 <- raster::setValues(imgBase, layer02)
  raster03 <- raster::setValues(imgBase, layer03)
  raster04 <- raster::setValues(imgBase, layer04)
  raster05 <- raster::setValues(imgBase, layer05)
  raster06 <- raster::setValues(imgBase, layer06)
  raster07 <- raster::setValues(imgBase, layer07)
  raster08 <- raster::setValues(imgBase, layer08)
  raster09 <- raster::setValues(imgBase, layer09)
  raster10 <- raster::setValues(imgBase, layer10)
  raster11 <- raster::setValues(imgBase, layer11)
  raster12 <- raster::setValues(imgBase, layer12)
  raster13 <- raster::setValues(imgBase, layer13)
  raster14 <- raster::setValues(imgBase, layer14)
  raster15 <- raster::setValues(imgBase, layer15)
  raster16 <- raster::setValues(imgBase, layer16)
  raster17 <- raster::setValues(imgBase, layer17)
  raster18 <- raster::setValues(imgBase, layer18)
  raster19 <- raster::setValues(imgBase, layer19)
  raster20 <- raster::setValues(imgBase, layer20)
  raster21 <- raster::setValues(imgBase, layer21)
  raster22 <- raster::setValues(imgBase, layer22)
  raster23 <- raster::setValues(imgBase, layer23)
  raster24 <- raster::setValues(imgBase, layer24)
  print(paste("[", phenoYears[currentYearIdx], "]", "rasters updated"))

  # Save
  phenologyGeotiffYearDir <- paste0(phenologyGeotiffDir, "/", phenoYears[currentYearIdx]) # creates /planet/phenologyGeotiff/site/year
  if (!dir.exists(phenologyGeotiffYearDir)) {
    dir.create(phenologyGeotiffYearDir)
    print(paste("[", phenoYears[currentYearIdx], "]", "created:", phenologyGeotiffYearDir))
  } else {
    print(paste("[", phenoYears[currentYearIdx], "]", "exists:", phenologyGeotiffYearDir))
  }

  raster::writeRaster(raster01, filename = paste0(phenologyGeotiffYearDir, "/01_", phenoYears[currentYearIdx], "_", productTable$short_name[1], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster02, filename = paste0(phenologyGeotiffYearDir, "/02_", phenoYears[currentYearIdx], "_", productTable$short_name[2], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster03, filename = paste0(phenologyGeotiffYearDir, "/03_", phenoYears[currentYearIdx], "_", productTable$short_name[3], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster04, filename = paste0(phenologyGeotiffYearDir, "/04_", phenoYears[currentYearIdx], "_", productTable$short_name[4], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster05, filename = paste0(phenologyGeotiffYearDir, "/05_", phenoYears[currentYearIdx], "_", productTable$short_name[5], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster06, filename = paste0(phenologyGeotiffYearDir, "/06_", phenoYears[currentYearIdx], "_", productTable$short_name[6], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster07, filename = paste0(phenologyGeotiffYearDir, "/07_", phenoYears[currentYearIdx], "_", productTable$short_name[7], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster08, filename = paste0(phenologyGeotiffYearDir, "/08_", phenoYears[currentYearIdx], "_", productTable$short_name[8], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster09, filename = paste0(phenologyGeotiffYearDir, "/09_", phenoYears[currentYearIdx], "_", productTable$short_name[9], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster10, filename = paste0(phenologyGeotiffYearDir, "/10_", phenoYears[currentYearIdx], "_", productTable$short_name[10], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster11, filename = paste0(phenologyGeotiffYearDir, "/11_", phenoYears[currentYearIdx], "_", productTable$short_name[11], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster12, filename = paste0(phenologyGeotiffYearDir, "/12_", phenoYears[currentYearIdx], "_", productTable$short_name[12], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster13, filename = paste0(phenologyGeotiffYearDir, "/13_", phenoYears[currentYearIdx], "_", productTable$short_name[13], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster14, filename = paste0(phenologyGeotiffYearDir, "/14_", phenoYears[currentYearIdx], "_", productTable$short_name[14], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster15, filename = paste0(phenologyGeotiffYearDir, "/15_", phenoYears[currentYearIdx], "_", productTable$short_name[15], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster16, filename = paste0(phenologyGeotiffYearDir, "/16_", phenoYears[currentYearIdx], "_", productTable$short_name[16], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster17, filename = paste0(phenologyGeotiffYearDir, "/17_", phenoYears[currentYearIdx], "_", productTable$short_name[17], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster18, filename = paste0(phenologyGeotiffYearDir, "/18_", phenoYears[currentYearIdx], "_", productTable$short_name[18], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster19, filename = paste0(phenologyGeotiffYearDir, "/19_", phenoYears[currentYearIdx], "_", productTable$short_name[19], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster20, filename = paste0(phenologyGeotiffYearDir, "/20_", phenoYears[currentYearIdx], "_", productTable$short_name[20], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster21, filename = paste0(phenologyGeotiffYearDir, "/21_", phenoYears[currentYearIdx], "_", productTable$short_name[21], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster22, filename = paste0(phenologyGeotiffYearDir, "/22_", phenoYears[currentYearIdx], "_", productTable$short_name[22], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster23, filename = paste0(phenologyGeotiffYearDir, "/23_", phenoYears[currentYearIdx], "_", productTable$short_name[23], ".tif"), format = "GTiff", overwrite = TRUE)
  raster::writeRaster(raster24, filename = paste0(phenologyGeotiffYearDir, "/24_", phenoYears[currentYearIdx], "_", productTable$short_name[24], ".tif"), format = "GTiff", overwrite = TRUE)

  print(paste(phenoYears[currentYearIdx], "complete"))
}
