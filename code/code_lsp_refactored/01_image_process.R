# adapted for PSScene where there is no longer a second UDM2 mask
# TODO
# based on this runs UDM1+UDM2 or UDM2 masking
# checks available files for a site
# rest of logic unchanged
# check character substr for dates,  file types

# OUTPUT
# is mosaic 4 band raster for each date in outDir with name format YYYYMMDD_cliped_mosaic.tif


# NOTE
# rgdal, gdalUtils, rgeos were archived from CRAN in 2023 in favor of sf/terra


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# A High Spatial Resolution Land Surface Phenology Dataset for AmeriFlux and NEON Sites
#
# 01: A script for PlanetScope image process
#
# Author: Minkyu Moon; moon.minkyu@gmail.com
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Submitted to the job for each site with shell script:
# #!/bin/bash
# echo Submitting $1
# R --vanilla < ~/01_img_process.R $1
#
# example submission command using default parameters:
# qsub -V -pe omp 28 -l h_rt=12:00:00 run_01.sh siteNumber
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

library(raster)
library(sp)
library(rjson)
library(geojsonR)
library(doMC)
library(doParallel)

########################################
args <- commandArgs()
print(args)

# siteNumber <- as.numeric(args[3])
siteNumber <- 103 # NOTE temp

########################################
## Load parameters

params <- fromJSON(file = "/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/PLSP_Parameters_refactored.json")
source(params$setup$rFunctions)

########################################
## Get site name,  image directory and coordinate
geojsonDir <- params$setup$geojsonDir

siteInfo <- GetSiteInfo(siteNumber, geojsonDir, params)

imgDir <- siteInfo[[1]] # in /raw
strSite <- siteInfo[[2]] # site name
print(paste(strSite, ";", imgDir))

cLong <- siteInfo[[3]]
cLat <- siteInfo[[4]]
print(paste(cLong, ";", cLat))

########################################
## Get list of files
# NOTE returns file names as of imgDir root, ex "data/0a1d0344-0a1d-4e57-8396-cafc75345c38/PSScene/20250625_183211_39_24b7_3B_AnalyticMS_SR_clip.tif"
fileNamesSR <- list.files(path = imgDir, pattern = glob2rx("*MS_SR*.tif"), recursive = TRUE)
fileNamesUDM <- list.files(path = imgDir, pattern = glob2rx("*_DN_udm*.tif"), recursive = TRUE)
fileNamesUDM2 <- list.files(path = imgDir, pattern = glob2rx("*_udm2*.tif"), recursive = TRUE)

# NOTE directory path is prepended to the file names
filePathsSR <- list.files(path = imgDir, pattern = glob2rx("*MS_SR*.tif"), recursive = TRUE, full.names = TRUE)
filePathsUDM <- list.files(path = imgDir, pattern = glob2rx("*_DN_udm*.tif"), recursive = TRUE, full.names = TRUE)
filePathsUDM2 <- list.files(path = imgDir, pattern = glob2rx("*_udm2*.tif"), recursive = TRUE, full.names = TRUE)

# strip path to accommodate PSScene4Band and PSScene
## Get dates
fileBasenamesSR <- basename(fileNamesSR)
fileBasenamesUDM <- basename(fileNamesUDM)
fileBasenamesUDM2 <- basename(fileNamesUDM2)


yy <- substr(fileBasenamesSR, 3, 4)
mm <- substr(fileBasenamesSR, 5, 6)
dd <- substr(fileBasenamesSR, 7, 8)
datesAll <- as.Date(paste(mm, dd, yy, sep = "/"), "%m/%d/%y")
uniqueDates <- unique(datesAll) # gets unique dates where data was downloaded

print(length(uniqueDates))


########################################
## Image process
## Set output directory for base image
outDir <- params$setup$outDir
outDirSite <- paste0(outDir, strSite)
if (!dir.exists(outDir)) {
  dir.create(outDir)
  print(paste("created:", outDir))
}
if (!dir.exists(outDirSite)) {
  dir.create(outDirSite)
  print(paste("created:", outDirSite))
}

## Create site shapefile and base image
siteWindow <- GetSiteShp(filePathsSR, cLong, cLat)
imgBase <- GetBaseImg(filePathsSR, siteWindow, outDirSite, save = TRUE)


##
registerDoMC(params$setup$numCores)

# Output directory for mosaic images
outDirSiteMosaic <- paste0(outDirSite, "/mosaic")
if (!dir.exists(outDirSiteMosaic)) {
  dir.create(outDirSiteMosaic)
  print(paste("created:", outDirSiteMosaic))
}

## Do a loop for each date
# foreach(currentDate = 1:length(uniqueDates)) %dopar% { # one parallel worker per date

currentDate <- 1

# Find images for current date
sameDateImages <- which(substr(fileBasenamesSR, 1, 8) == paste0(substr(uniqueDates[currentDate], 1, 4), substr(uniqueDates[currentDate], 6, 7), substr(uniqueDates[currentDate], 9, 10)))

# Find images that have all 4 PlanetScope bands
valid4BandImages <- c()
for (currentSameDateImageIdx in 1:length(sameDateImages)) {
  log <- try(
    {
      img <- raster::raster(filePathsSR[sameDateImages[currentSameDateImageIdx]])
      img <- raster::crop(img, siteWindow) # crop scenes to site geojson
    },
    silent = TRUE
  ) # log is of type try-error, if img fails error is suppressed
  if (inherits(log, "try-error")) {
    next
  } else {
    numBand <- raster::nbands(raster::raster(filePathsSR[sameDateImages[currentSameDateImageIdx]]))
    if (numBand == 4) {
      valid4BandImages <- c(valid4BandImages, sameDateImages[currentSameDateImageIdx])
    } else {
      print(paste("doesn't have 4 bands", currentDate, ",", currentSameDateImageIdx))
    }
  }
}

# If the number of images that have 4 bands is more than zero,  load them and create a mosaic image
if (length(valid4BandImages) > 0) {
  imgB <- vector("list", length(valid4BandImages))

  for (currentValidImageIdx in 1:length(valid4BandImages)) { # get each individual scene
    ii <- valid4BandImages[currentValidImageIdx]

    img <- raster::raster(filePathsSR[ii])
    numBand <- raster::nbands(img)

    # get scene id to find paired UDM files
    currentSceneId <- substr(fileBasenamesSR[ii], 1, unlist(gregexpr("3B", fileBasenamesSR[ii])) - 2)
    currentSceneIdLength <- nchar(currentSceneId)
    # crop up to 3B then remove
    # ex1 files
    # 20210526_152814_104b_3B_AnalyticMS_DN_udm.tif
    # 	20210526_152814_104b_3B_AnalyticMS_SR.tif
    # 	20210526_152814_104b_3B_udm2.tif
    # ex2 files
    # 20250625_183211_39_24b7_3B_AnalyticMS_SR_clip.tif
    # 20250625_183211_39_24b7_3B_udm2_clip.tif


    imgP <- vector("list", numBand)
    for (currentBand in 1:numBand) {
      imgT <- raster::raster(filePathsSR[ii], band = currentBand)
      imgT <- raster::crop(imgT, siteWindow)

      if (length(which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)) == 1 & length(which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)) == 0) {
        # only UDM quality mask exists
        log <- try(
          {
            udmT <- raster::raster(filePathsUDM[which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)])
            udmT <- raster::crop(udmT, siteWindow)
          },
          silent = TRUE
        )
        if (inherits(log, "try-error")) {
          next
        } else {
          imgT[udmT > 0] <- NA
        }
      } else if (length(which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)) == 1 & length(which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)) == 1) {
        # use UDM1 and UDM 2quality mask
        log <- try(
          {
            udmT <- raster::raster(filePathsUDM[which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)])
            udmT <- raster::crop(udmT, siteWindow)
          },
          silent = TRUE
        )
        if (inherits(log, "try-error")) {
          imgT <- imgT
        } else {
          imgT[udmT > 0] <- NA
        }
        log <- try(
          {
            udm2T <- raster(filePathsUDM2[which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)])
            udm2T <- crop(udm2T, siteWindow)
          },
          silent = TRUE
        )
        if (inherits(log, "try-error")) {
          imgT <- imgT
        } else {
          imgT[udm2T != 1] <- NA
        }
      }
      imgP[[currentBand]] <- imgT # current band masked raster
    }

    imgB[[currentValidImageIdx]] <- brick(imgP) # builds 4 band image for scene
  }

  temp1 <- vector("list", (length(valid4BandImages) + 1))
  temp2 <- vector("list", (length(valid4BandImages) + 1))
  temp3 <- vector("list", (length(valid4BandImages) + 1))
  temp4 <- vector("list", (length(valid4BandImages) + 1))
  for (i in 1:length(valid4BandImages)) {
    temp1[[i]] <- raster::raster(imgB[[i]], 1) # band 1
    temp2[[i]] <- raster::raster(imgB[[i]], 2) # band 2
    temp3[[i]] <- raster::raster(imgB[[i]], 3) # band 3
    temp4[[i]] <- raster::raster(imgB[[i]], 4) # band 4
  }
  temp1[[(length(valid4BandImages) + 1)]] <- imgBase
  temp2[[(length(valid4BandImages) + 1)]] <- imgBase
  temp3[[(length(valid4BandImages) + 1)]] <- imgBase
  temp4[[(length(valid4BandImages) + 1)]] <- imgBase

  # Check their spatial information
  for (i in 1:length(valid4BandImages)) {
    log <- try(raster::compareRaster(temp1[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
    if (inherits(log, "try-error")) {
      temp1[[i]] <- raster::projectRaster(temp1[[i]], imgBase)
    }

    log <- try(raster::compareRaster(temp2[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
    if (inherits(log, "try-error")) {
      temp2[[i]] <- raster::projectRaster(temp2[[i]], imgBase)
    }

    log <- try(raster::compareRaster(temp3[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
    if (inherits(log, "try-error")) {
      temp3[[i]] <- raster::projectRaster(temp3[[i]], imgBase)
    }

    log <- try(raster::compareRaster(temp4[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
    if (inherits(log, "try-error")) {
      temp4[[i]] <- raster::projectRaster(temp4[[i]], imgBase)
    }
  }
  temp1$fun <- mean # add function mean
  temp2$fun <- mean
  temp3$fun <- mean
  temp4$fun <- mean
  temp1$na.rm <- TRUE
  temp2$na.rm <- TRUE
  temp3$na.rm <- TRUE
  temp4$na.rm <- TRUE
  # for mosaic, fun and na.rm are named arguments controlling how overlapping pixels are combined
  rasterBand1 <- do.call(raster::mosaic, temp1) # (function,  list of arguments),  calls mosaic on each raster individually
  rasterBand2 <- do.call(raster::mosaic, temp2) # do.call() allows for variable number of arguments
  rasterBand3 <- do.call(raster::mosaic, temp3)
  rasterBand4 <- do.call(raster::mosaic, temp4)

  # Brick bands
  combined4BandRaster <- raster::brick(rasterBand1, rasterBand2, rasterBand3, rasterBand4)

  # Save
  outFile <- paste0(outDirSiteMosaic, "/", substr(uniqueDates[currentDate], 1, 4), substr(uniqueDates[currentDate], 6, 7), substr(uniqueDates[currentDate], 9, 10), "_cliped_mosaic.tif")
  writeRaster(combined4BandRaster, filename = outFile, format = "GTiff", overwrite = TRUE)

  print(paste("saved:", outFile))
}
# }

# Check the length of output - number of unique mosaics and number of unique dates
print(paste("number of mosaics created:", length(list.files(path = outDirSiteMosaic))))
print(paste("number of unique dates:", length(uniqueDates)))


# TODO check masking, clouds still appear
