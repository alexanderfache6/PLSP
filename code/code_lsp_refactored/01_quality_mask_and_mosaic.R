# adapted for PSScene where there is no longer a second UDM2 mask
# TODO
# based on this runs UDM1+UDM2 or UDM2 masking
# checks available files for a site
# rest of logic unchanged
# check character substr for dates,  file types

# OUTPUT
# is mosaic 4 band raster for each date in outputDir with name format YYYYMMDD_cliped_mosaic.tif


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
# library(doMC) # Provides a parallel backend for the %dopar% function using the multicore functionality of the parallel package.
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

print(paste("length(uniqueDates):", length(uniqueDates)))


########################################
## Image process
## Set output directory for base image
outputDir <- params$setup$outDir
outputDirSite <- paste0(outputDir, strSite)
if (!dir.exists(outputDir)) {
  dir.create(outputDir)
  print(paste("created:", outputDir))
}
if (!dir.exists(outputDirSite)) {
  dir.create(outputDirSite)
  print(paste("created:", outputDirSite))
}

## Create site shapefile and base image
siteWindow <- GetSiteShp(filePathsSR, cLong, cLat)
imgBase <- GetBaseImg(filePathsSR, siteWindow, outputDirSite, save = TRUE)




# Output directory for mosaic images
outputDirSiteMosaic <- paste0(outputDirSite, "/mosaic")
if (!dir.exists(outputDirSiteMosaic)) {
  dir.create(outputDirSiteMosaic)
  print(paste("created:", outputDirSiteMosaic))
}


# registerDoMC(cores = params$setup$numCores) # register the multicore parallel backend with the foreach package
cluster <- makeCluster(params$setup$numCores)
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(raster)
  library(sp)
})

clusterExport(cluster, varlist = c(
  "filePathsSR", "filePathsUDM", "filePathsUDM2",
  "fileBasenamesSR", "fileBasenamesUDM", "fileBasenamesUDM2",
  "uniqueDates", "imgBase", "siteWindow", "outputDirSiteMosaic"
))


# runs one parallel worker per date
foreach(currentDate = 1:length(uniqueDates)) %dopar% {
# foreach(currentDate = 1:5) %dopar% {
  # Find images for current date
  currentDateStr <- paste0(substr(uniqueDates[currentDate], 1, 4), substr(uniqueDates[currentDate], 6, 7), substr(uniqueDates[currentDate], 9, 10))
  print(paste("currentDateStr:", currentDateStr))

  sameDateImages <- which(substr(fileBasenamesSR, 1, 8) == currentDateStr)
  print(paste("length(sameDateImages):", length(sameDateImages)))

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

  # If the number of images that have 4 bands is more than zero, load them and create a mosaic image
  print(paste(length(valid4BandImages), "/", length(sameDateImages), "valid 4Band images"))


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


      # NOTE check masking, clouds still appear
      # this was because there was no UDM1=0 & UDM2=1 condition, new images don't have UDM1


      imgP <- vector("list", numBand)
      for (currentBand in 1:numBand) {
        print(paste("currentSceneId:", currentSceneId, "| currentBand:", currentBand))
        imgT <- raster::raster(filePathsSR[ii], band = currentBand)
        imgT <- raster::crop(imgT, siteWindow)

        if (length(which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)) == 1 & length(which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)) == 0) {
          print("UDM1 = 1, UDM2 = 0")
          # only UDM quality mask exists
          log <- try(
            {
              udmT <- raster::raster(filePathsUDM[which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)]) # load raster
              udmT <- raster::crop(udmT, siteWindow) # crop to site extent
            },
            silent = TRUE
          )
          if (inherits(log, "try-error")) {
            next # if error in quality mask, skip image
          } else {
            imgT[udmT > 0] <- NA # A value of zero indicates a "good" imagery pixel
            # values > 0 are unusable, set to NA
          }
        } else if (length(which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)) == 1 & length(which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)) == 1) {
          print("UDM1 = 1, UDM2 = 1")
          # UDM and UDM2 quality mask exist
          log <- try(
            {
              udmT <- raster::raster(filePathsUDM[which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)])
              udmT <- raster::crop(udmT, siteWindow)
            },
            silent = TRUE
          )
          if (inherits(log, "try-error")) {
            imgT <- imgT # don't skip if issue in UDM1, check UDM1 next
          } else {
            imgT[udmT > 0] <- NA
          }
          log <- try(
            {
              udm2T <- raster::raster(filePathsUDM2[which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)])
              udm2T <- raster::crop(udm2T, siteWindow)
            },
            silent = TRUE
          )
          if (inherits(log, "try-error")) {
            imgT <- imgT # TODO if error don't modify image, inherit from UDM1. what if both have an issue?? this is not addressed
          } else {
            imgT[udm2T != 1] <- NA # set imgT pixels to NA if the udm2T mask != 1
            # udm2T == 1 means the pixel has a clear sky https://docs.planet.com/data/imagery/udm/#udm21-product-bands
          }
        } else if (length(which(substr(fileBasenamesUDM, 1, currentSceneIdLength) == currentSceneId)) == 0 & length(which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)) == 1) {
          print("UDM1 = 0, UDM2 = 1")
          log <- try(
            {
              udmT <- raster::raster(filePathsUDM2[which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)], band = 8) # band 8 holds old UDM mask when UDM1 doesn't exist
              udmT <- raster::crop(udmT, siteWindow)
            },
            silent = TRUE
          )
          if (inherits(log, "try-error")) {
            next
          } else {
            imgT[udmT > 0] <- NA
          }
          log <- try(
            {
              udm2T <- raster::raster(filePathsUDM2[which(substr(fileBasenamesUDM2, 1, currentSceneIdLength) == currentSceneId)])
              udm2T <- raster::crop(udm2T, siteWindow)
            },
            silent = TRUE
          )
          if (inherits(log, "try-error")) {
            imgT <- imgT
          } else {
            imgT[udm2T != 1] <- NA
          }
        } else {
          print("UDM1 = 0, UDM2 = 0")
          print("---------- ERROR no quality mask applied ----------")
        }

        imgP[[currentBand]] <- imgT # add current band raster that has been quality checked
      }

      imgB[[currentValidImageIdx]] <- raster::brick(imgP) # builds 4 band image for scene
    }

    validImagesBand1 <- vector("list", (length(valid4BandImages) + 1))
    validImagesBand2 <- vector("list", (length(valid4BandImages) + 1))
    validImagesBand3 <- vector("list", (length(valid4BandImages) + 1))
    validImagesBand4 <- vector("list", (length(valid4BandImages) + 1))
    for (i in 1:length(valid4BandImages)) { # separate individual bands and combine across valid images
      validImagesBand1[[i]] <- raster::raster(imgB[[i]], 1) # band 1
      validImagesBand2[[i]] <- raster::raster(imgB[[i]], 2) # band 2
      validImagesBand3[[i]] <- raster::raster(imgB[[i]], 3) # band 3
      validImagesBand4[[i]] <- raster::raster(imgB[[i]], 4) # band 4
    }
    validImagesBand1[[(length(valid4BandImages) + 1)]] <- imgBase # TODO why add base image, I think fallback if no pixels for a particular date??
    validImagesBand2[[(length(valid4BandImages) + 1)]] <- imgBase
    validImagesBand3[[(length(valid4BandImages) + 1)]] <- imgBase
    validImagesBand4[[(length(valid4BandImages) + 1)]] <- imgBase

    # Check their spatial information
    for (i in 1:length(valid4BandImages)) {
      log <- try(raster::compareRaster(validImagesBand1[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE) # have the same crs, resolutions
      if (inherits(log, "try-error")) {
        validImagesBand1[[i]] <- raster::projectRaster(validImagesBand1[[i]], imgBase)
      }

      log <- try(raster::compareRaster(validImagesBand2[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        validImagesBand2[[i]] <- raster::projectRaster(validImagesBand2[[i]], imgBase)
      }

      log <- try(raster::compareRaster(validImagesBand3[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        validImagesBand3[[i]] <- raster::projectRaster(validImagesBand3[[i]], imgBase)
      }

      log <- try(raster::compareRaster(validImagesBand4[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        validImagesBand4[[i]] <- raster::projectRaster(validImagesBand4[[i]], imgBase)
      }
    }
    # prep bands for aggregation, add mean function, and na.rm
    validImagesBand1$fun <- mean
    validImagesBand2$fun <- mean
    validImagesBand3$fun <- mean
    validImagesBand4$fun <- mean
    validImagesBand1$na.rm <- TRUE
    validImagesBand2$na.rm <- TRUE
    validImagesBand3$na.rm <- TRUE
    validImagesBand4$na.rm <- TRUE
    # for mosaic, fun and na.rm are named arguments controlling how overlapping pixels are combined
    rasterBand1 <- do.call(raster::mosaic, validImagesBand1) # (function,  list of arguments),  calls mosaic on each raster individually
    rasterBand2 <- do.call(raster::mosaic, validImagesBand2) # do.call() allows for variable number of arguments
    rasterBand3 <- do.call(raster::mosaic, validImagesBand3)
    rasterBand4 <- do.call(raster::mosaic, validImagesBand4)

    # Brick bands ie create multi layer raster object
    combined4BandRaster <- raster::brick(rasterBand1, rasterBand2, rasterBand3, rasterBand4)

    # Save
    outFile <- paste0(outputDirSiteMosaic, "/", substr(uniqueDates[currentDate], 1, 4), substr(uniqueDates[currentDate], 6, 7), substr(uniqueDates[currentDate], 9, 10), "_clipped_mosaic.tif")
    raster::writeRaster(combined4BandRaster, filename = outFile, format = "GTiff", overwrite = TRUE)

    print(paste("saved:", outFile))
  }
} # end %dopar%

stopCluster(cluster)

# Check the length of output
# number of unique mosaics should equal number of unique dates (1 mosaic per day)
print(paste("number of mosaics created:", length(list.files(path = outputDirSiteMosaic, pattern = glob2rx("*_clipped_mosaic.tif")))))
print(paste("number of unique dates:", length(uniqueDates)))
