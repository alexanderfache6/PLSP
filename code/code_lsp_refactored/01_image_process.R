# adapted for PSScene where there is no longer a second UDM2 mask
# TODO
# based on this runs UDM1+UDM2 or UDM2 masking
# checks available files for a site
# rest of logic unchanged
# check character substr for dates,  file types

# OUTPUT
# is mosaic 4 band raster for each date in outDir with name format YYYYMMDD_cliped_mosaic.tif


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
library(rgdal)
library(gdalUtils)
library(rgeos)

library(rjson)
library(geojsonR)

library(doMC)
library(doParallel)

########################################
args <- commandArgs()
print(args)

siteNumber <- as.numeric(args[3])

########################################
## Load parameters
params <- fromJSON(file = "~/PLSP_Parameters_refactored.json") # NOTE updated
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
# NOTE returns file names
fileNameSR <- list.files(path = imgDir, pattern = glob2rx("*MS_SR*.tif"), recursive = TRUE)
fileNameUDM <- list.files(path = imgDir, pattern = glob2rx("*_DN_udm*.tif"), recursive = TRUE)
fileNameUDM2 <- list.files(path = imgDir, pattern = glob2rx("*_udm2*.tif"), recursive = TRUE)

# NOTE directory path is prepended to the file names
filePathSR <- list.files(path = imgDir, pattern = glob2rx("*MS_SR*.tif"), recursive = TRUE, full.names = TRUE)
filePathUDM <- list.files(path = imgDir, pattern = glob2rx("*_DN_udm*.tif"), recursive = TRUE, full.names = TRUE)
filePathUDM2 <- list.files(path = imgDir, pattern = glob2rx("*_udm2*.tif"), recursive = TRUE, full.names = TRUE)


# TODO check date character locations
## Get dates
yy <- substr(fileNameSR, 58, 59)
mm <- substr(fileNameSR, 60, 61)
dd <- substr(fileNameSR, 62, 63)
datesAll <- as.Date(paste(mm, "/", dd, "/", yy, sep = ""), "%m/%d/%y")
uniqueDates <- unique(datesAll) # gets unique dates where data was downloaded

print(length(uniqueDates))


########################################
## Image process
## Set output directory for base image
outDir <- paste0(params$setup$outDir, strSite)
if (!dir.exists(outDir)) {
  dir.create(outDir)
}

## Create site shapefile and base image
siteWindow <- GetSiteShp(filePathSR, cLong, cLat)
imgBase <- GetBaseImg(filePathSR, siteWindow, outDir, save = TRUE)


##
registerDoMC(params$setup$numCores)

# Output directory for mosaic images
outDir <- paste0(params$setup$outDir, strSite, "/mosaic")
if (!dir.exists(outDir)) {
  dir.create(outDir)
}

## Do a loop for each date
foreach(current_date = 1:length(uniqueDates)) %dopar% { # one parallel worker per date

  # Find images for a date
  sameDateImages <- which(substr(fileNameSR, 56, 63) == paste0(substr(uniqueDates[current_date], 1, 4), substr(uniqueDates[current_date], 6, 7), substr(uniqueDates[current_date], 9, 10)))

  # Find images that have all 4 PlanetScope bands
  valid4BandImages <- c()
  for (mm in 1:length(sameDateImages)) {
    log <- try(
      {
        img <- raster(filePathSR[sameDateImages[mm]])
        img <- crop(img, siteWindow) # crop scenes to site geojson
      },
      silent = TRUE
    ) # log is of type try-error, if img fails error is suppressed
    if (inherits(log, "try-error")) {
      next
    } else {
      numBand <- nbands(raster(filePathSR[sameDateImages[mm]]))
      if (numBand == 4) {
        valid4BandImages <- c(valid4BandImages, sameDateImages[mm])
      }
    }
  }

  # If the number of images that have 4 bands is more than zero,  load them and create a mosaic image
  if (length(valid4BandImages) > 0) {
    imgB <- vector("list", length(valid4BandImages))

    for (mm in 1:length(valid4BandImages)) { # get each individual scene
      ii <- valid4BandImages[mm]

      img <- raster(filePathSR[ii])
      numBand <- nbands(img)

      str <- substr(fileNameSR[ii], 56, 77) # get scene id

      imgP <- vector("list", numBand)
      for (i in 1:numBand) {
        imgT <- raster(filePathSR[ii], band = i)
        imgT <- crop(imgT, siteWindow)

        if (length(which(substr(fileNameUDM, 56, 77) == str)) == 1 & length(which(substr(fileNameUDM2, 56, 77) == str)) == 0) {
          # use UDM1 and UDM 2 quality masks
          log <- try(
            {
              udmT <- raster(filePathUDM[which(substr(fileNameUDM, 56, 77) == str)])
              udmT <- crop(udmT, siteWindow)
            },
            silent = TRUE
          )
          if (inherits(log, "try-error")) {
            next
          } else {
            imgT[udmT > 0] <- NA
          }
        } else if (length(which(substr(fileNameUDM, 56, 77) == str)) == 1 & length(which(substr(fileNameUDM2, 56, 77) == str)) == 1) {
          # use UDM1 quality mask
          log <- try(
            {
              udmT <- raster(filePathUDM[which(substr(fileNameUDM, 56, 77) == str)])
              udmT <- crop(udmT, siteWindow)
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
              udm2T <- raster(filePathUDM2[which(substr(fileNameUDM2, 56, 77) == str)])
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
        imgP[[i]] <- imgT # current band masked raster
      }

      imgB[[mm]] <- brick(imgP) # builds 4 band image for scene
    }

    temp1 <- vector("list", (length(valid4BandImages) + 1))
    temp2 <- vector("list", (length(valid4BandImages) + 1))
    temp3 <- vector("list", (length(valid4BandImages) + 1))
    temp4 <- vector("list", (length(valid4BandImages) + 1))
    for (i in 1:length(valid4BandImages)) {
      temp1[[i]] <- raster(imgB[[i]], 1)
      temp2[[i]] <- raster(imgB[[i]], 2)
      temp3[[i]] <- raster(imgB[[i]], 3)
      temp4[[i]] <- raster(imgB[[i]], 4)
    }
    temp1[[(length(valid4BandImages) + 1)]] <- imgBase
    temp2[[(length(valid4BandImages) + 1)]] <- imgBase
    temp3[[(length(valid4BandImages) + 1)]] <- imgBase
    temp4[[(length(valid4BandImages) + 1)]] <- imgBase

    # Check their spatial information
    for (i in 1:length(valid4BandImages)) {
      log <- try(compareRaster(temp1[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        temp1[[i]] <- projectRaster(temp1[[i]], imgBase)
      }
      log <- try(compareRaster(temp2[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        temp2[[i]] <- projectRaster(temp2[[i]], imgBase)
      }
      log <- try(compareRaster(temp3[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        temp3[[i]] <- projectRaster(temp3[[i]], imgBase)
      }
      log <- try(compareRaster(temp4[[i]], imgBase, extent = FALSE, rowcol = FALSE), silent = TRUE)
      if (inherits(log, "try-error")) {
        temp4[[i]] <- projectRaster(temp4[[i]], imgBase)
      }
    }
    temp1$fun <- mean
    temp2$fun <- mean
    temp3$fun <- mean
    temp4$fun <- mean # add function mean
    temp1$na.rm <- TRUE
    temp2$na.rm <- TRUE
    temp3$na.rm <- TRUE
    temp4$na.rm <- TRUE
    # for mosaic,  fun and na.rm as named arguments controlling how overlapping pixels are combined
    rasterBand1 <- do.call(mosaic, temp1) # (function,  list of arguments),  calls mosaic on each raster individually
    rasterBand2 <- do.call(mosaic, temp2) # do.call() allows for variable number of arguments
    rasterBand3 <- do.call(mosaic, temp3)
    rasterBand4 <- do.call(mosaic, temp4)

    # Brick bands
    combined4BandRaster <- brick(rasterBand1, rasterBand2, rasterBand3, rasterBand4)

    # Save
    outFile <- paste0(outDir, "/", substr(uniqueDates[current_date], 1, 4), substr(uniqueDates[current_date], 6, 7), substr(uniqueDates[current_date], 9, 10), "_cliped_mosaic.tif")
    writeRaster(combined4BandRaster, filename = outFile, format = "GTiff", overwrite = TRUE)

    print(outFile)
  }
}

# Check the length of output
print(length(list.files(path = outDir)))

print(length(uniqueDates))
