# NOTE creates 24 layer/metric tif phenologyChunkFiles per site per year


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
  l01 <- matrix(NA, numberOfPixels, 1)
  l02 <- matrix(NA, numberOfPixels, 1)
  l03 <- matrix(NA, numberOfPixels, 1)
  l04 <- matrix(NA, numberOfPixels, 1)
  l05 <- matrix(NA, numberOfPixels, 1)
  l06 <- matrix(NA, numberOfPixels, 1)
  l07 <- matrix(NA, numberOfPixels, 1)
  l08 <- matrix(NA, numberOfPixels, 1)
  l09 <- matrix(NA, numberOfPixels, 1)
  l10 <- matrix(NA, numberOfPixels, 1)
  l11 <- matrix(NA, numberOfPixels, 1)
  l12 <- matrix(NA, numberOfPixels, 1)
  l13 <- matrix(NA, numberOfPixels, 1)
  l14 <- matrix(NA, numberOfPixels, 1)
  l15 <- matrix(NA, numberOfPixels, 1)
  l16 <- matrix(NA, numberOfPixels, 1)
  l17 <- matrix(NA, numberOfPixels, 1)
  l18 <- matrix(NA, numberOfPixels, 1)
  l19 <- matrix(NA, numberOfPixels, 1)
  l20 <- matrix(NA, numberOfPixels, 1)
  l21 <- matrix(NA, numberOfPixels, 1)
  l22 <- matrix(NA, numberOfPixels, 1)
  l23 <- matrix(NA, numberOfPixels, 1)
  l24 <- matrix(NA, numberOfPixels, 1)

  for (currentChunk in 1:numberOfChunks) {
    currentChunkFile <- paste0(phenologyChunkDir, "/chunk_phenology_", currentChunkStringified, ".rda")
    log <- try(load(currentChunkFile, verbose = TRUE), silent = FALSE)
    # loads phenoResultsMatrix from 03_run_LSP.R
    if (inherits(log, "try-error")) next

    if (currentChunk == numberOfChunks) {
      chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):length(imgBase)) # at last chunk, include remainder of pixels
    } else {
      chunkPixelIndices <- c((chunkSize * (currentChunk - 1) + 1):(chunkSize * currentChunk))
    }

    chunkStart <- chunkPixelIndices[1]
    chunkEnd <- chunkPixelIndices[length(chunkPixelIndices)]

    l01[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 1)]
    l02[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 2)]
    l03[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 3)]
    l04[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 4)]
    l05[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 5)]
    l06[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 6)]
    l07[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 7)]
    l08[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 8)]
    l09[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 9)]
    l10[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 10)]
    l11[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 11)]
    l12[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 12)]
    l13[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 13)]
    l14[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 14)]
    l15[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 15)]
    l16[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 16)]
    l17[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 17)]
    l18[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 18)]
    l19[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 19)]
    l20[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 20)]
    l21[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 21)]
    l22[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 22)]
    l23[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 23)]
    l24[chunkStart:chunkEnd, ] <- phenoResultsMatrix[, (24 * (currentYearIdx - 1) + 24)]
  }

  r01 <- setValues(imgBase, l01)
  r02 <- setValues(imgBase, l02)
  r03 <- setValues(imgBase, l03)
  r04 <- setValues(imgBase, l04)
  r05 <- setValues(imgBase, l05)
  r06 <- setValues(imgBase, l06)
  r07 <- setValues(imgBase, l07)
  r08 <- setValues(imgBase, l08)
  r09 <- setValues(imgBase, l09)
  r10 <- setValues(imgBase, l10)
  r11 <- setValues(imgBase, l11)
  r12 <- setValues(imgBase, l12)
  r13 <- setValues(imgBase, l13)
  r14 <- setValues(imgBase, l14)
  r15 <- setValues(imgBase, l15)
  r16 <- setValues(imgBase, l16)
  r17 <- setValues(imgBase, l17)
  r18 <- setValues(imgBase, l18)
  r19 <- setValues(imgBase, l19)
  r20 <- setValues(imgBase, l20)
  r21 <- setValues(imgBase, l21)
  r22 <- setValues(imgBase, l22)
  r23 <- setValues(imgBase, l23)
  r24 <- setValues(imgBase, l24)


  # Save
  phenologyGeotiffYearDir <- paste0(phenologyGeotiffDir, "/", phenoYears[currentYearIdx]) # creates /planet/phenologyGeotiff/site/year
  if (!dir.exists(phenologyGeotiffYearDir)) {
    dir.create(phenologyGeotiffYearDir)
    print(paste("created:", phenologyGeotiffYearDir))
  } else {
    print(paste("exists:", phenologyGeotiffYearDir))
  }

  writeRaster(r01, filename = paste0(phenologyGeotiffYearDir, "/01_", phenoYears[currentYearIdx], "_", productTable$short_name[1], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r02, filename = paste0(phenologyGeotiffYearDir, "/02_", phenoYears[currentYearIdx], "_", productTable$short_name[2], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r03, filename = paste0(phenologyGeotiffYearDir, "/03_", phenoYears[currentYearIdx], "_", productTable$short_name[3], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r04, filename = paste0(phenologyGeotiffYearDir, "/04_", phenoYears[currentYearIdx], "_", productTable$short_name[4], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r05, filename = paste0(phenologyGeotiffYearDir, "/05_", phenoYears[currentYearIdx], "_", productTable$short_name[5], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r06, filename = paste0(phenologyGeotiffYearDir, "/06_", phenoYears[currentYearIdx], "_", productTable$short_name[6], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r07, filename = paste0(phenologyGeotiffYearDir, "/07_", phenoYears[currentYearIdx], "_", productTable$short_name[7], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r08, filename = paste0(phenologyGeotiffYearDir, "/08_", phenoYears[currentYearIdx], "_", productTable$short_name[8], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r09, filename = paste0(phenologyGeotiffYearDir, "/09_", phenoYears[currentYearIdx], "_", productTable$short_name[9], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r10, filename = paste0(phenologyGeotiffYearDir, "/10_", phenoYears[currentYearIdx], "_", productTable$short_name[10], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r11, filename = paste0(phenologyGeotiffYearDir, "/11_", phenoYears[currentYearIdx], "_", productTable$short_name[11], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r12, filename = paste0(phenologyGeotiffYearDir, "/12_", phenoYears[currentYearIdx], "_", productTable$short_name[12], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r13, filename = paste0(phenologyGeotiffYearDir, "/13_", phenoYears[currentYearIdx], "_", productTable$short_name[13], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r14, filename = paste0(phenologyGeotiffYearDir, "/14_", phenoYears[currentYearIdx], "_", productTable$short_name[14], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r15, filename = paste0(phenologyGeotiffYearDir, "/15_", phenoYears[currentYearIdx], "_", productTable$short_name[15], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r16, filename = paste0(phenologyGeotiffYearDir, "/16_", phenoYears[currentYearIdx], "_", productTable$short_name[16], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r17, filename = paste0(phenologyGeotiffYearDir, "/17_", phenoYears[currentYearIdx], "_", productTable$short_name[17], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r18, filename = paste0(phenologyGeotiffYearDir, "/18_", phenoYears[currentYearIdx], "_", productTable$short_name[18], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r19, filename = paste0(phenologyGeotiffYearDir, "/19_", phenoYears[currentYearIdx], "_", productTable$short_name[19], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r20, filename = paste0(phenologyGeotiffYearDir, "/20_", phenoYears[currentYearIdx], "_", productTable$short_name[20], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r21, filename = paste0(phenologyGeotiffYearDir, "/21_", phenoYears[currentYearIdx], "_", productTable$short_name[21], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r22, filename = paste0(phenologyGeotiffYearDir, "/22_", phenoYears[currentYearIdx], "_", productTable$short_name[22], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r23, filename = paste0(phenologyGeotiffYearDir, "/23_", phenoYears[currentYearIdx], "_", productTable$short_name[23], ".tif"), format = "GTiff", overwrite = TRUE)
  writeRaster(r24, filename = paste0(phenologyGeotiffYearDir, "/24_", phenoYears[currentYearIdx], "_", productTable$short_name[24], ".tif"), format = "GTiff", overwrite = TRUE)

  print(paste(phenoYears[currentYearIdx], "complete"))
}
