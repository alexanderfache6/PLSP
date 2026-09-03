# USAGE just run manually in RStudio with 4core like 04
# NOTE create single year PLSP_.nc file with all metrics and metadata


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# A High Spatial Resolution Land Surface Phenology Dataset for AmeriFlux and NEON Sites
#
# 05: A script for saving data layers into netCDF format
#
# Author: Minkyu Moon; moon.minkyu@gmail.com
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Submitted to the job for a single chunk for a site with shell script:
# #!/bin/bash
# echo Submitting $1
# R --vanilla < ~/05_netCDF.R $1
#
# example submission command using default parameters:
# qsub -V -pe omp 2 -l h_rt=12:00:00 run_05.sh numSite
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

library(sp) # CRS
library(terra)
library(sf) # wkt
library(ncdf4)
library(rjson)
library(geojsonR) # FROM_GeoJson()

########################################
# args <- commandArgs()
# print(args)

# siteNumber <- as.numeric(args[4])
siteNumber <- 75 # NOTE temp when running in RStudio
print(paste("siteNumber:", siteNumber))

########################################
## Load parameters
paramsFile <- "/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/PLSP_Parameters_refactored.json"
params <- fromJSON(file = paramsFile)
source(params$setup$rFunctions)

print("========================================")
print(paste("[PLSP_Parameters_refactored.json] file:", paramsFile))
print(readLines(paramsFile, warn = FALSE)) # NOTE suppresses missing blank line in json
print("========================================")

########################################
inBase <- params$setup$phenologyGeotiffDir
outBase <- params$setup$netCDFStageDir
print(paste("inBase:", inBase))
print(paste("outBase:", outBase))

if (!dir.exists(outBase)) {
  dir.create(outBase)
  print(paste("created:", outBase))
} else {
  print(paste("exists:", outBase))
}

geojsonDir <- params$setup$geojsonDir

siteInfo <- GetSiteInfo(siteNumber, geojsonDir, params)
strSite <- siteInfo[[2]] # site name
print(paste("strSite:", strSite))

########################################
# Get product layers info
productTable <- read.csv("/projectnb/modislc/users/fache/src/PLSP/PLSP_Layers.csv", header = TRUE, stringsAsFactors = FALSE)
# Get a base image to pull raster info from
baseImage <- raster::raster(paste0(params$setup$mosaicsDir, strSite, "/base_image.tif"))

########################################
# Get extent, and then define pixel centers in the x and y direction
base_extent <- raster::extent(baseImage)
res <- raster::res(baseImage)[1]
x <- seq(base_extent[1] + res / 2, base_extent[2] - res / 2, res)
y <- seq(base_extent[3] + res / 2, base_extent[4] - res / 2, res)
# Define dimensions for netCDF file
dimx <- ncdf4::ncdim_def(name = "x", longname = "x coordinate", units = "m", vals = as.double(x))
dimy <- ncdf4::ncdim_def(name = "y", longname = "y coordinate", units = "m", vals = rev(as.double(y)))


########################################
# loop through phenology years to create nc files

phenoYears <- params$setup$phenStartYr:params$setup$phenEndYr
print(paste("phenoYears:", paste(phenoYears, collapse = ", ")))

for (currentYear in phenoYears) {
  print(paste("[", currentYear, "]", "started"))
  
  # Load files
  files <- list.files(paste0(inBase, strSite, "/", currentYear), pattern = glob2rx(paste("*.tif", sep = "")), full.names = TRUE)

  outFold <- paste0(outBase, strSite, "/")
  if (!dir.exists(outFold)) {
    dir.create(outFold)
    print(paste("created:", outFold))
  } else {
    print(paste("exists:", outFold))
  }
  outFile <- paste0(outFold, "PLSP_", currentYear, ".nc")

  # Loop through all the layers, and create a variable for each
  results <- vector("list", dim(productTable)[1] + 1)
  results[[1]] <- ncdf4::ncvar_def("transverse_mercator", "", list(), prec = "char")

  print(paste("[", currentYear, "]", "creating nc variables"))
  for (i in 1:dim(productTable)[1]) {
    currentLayer <- productTable[i, ] # Pull the info for this layer from the productTable

    if (currentLayer$data_type == "Int16") {
      precision <- "short"
    } # All are int16, so this isn't necessary

    # Create the variable, add to the list. Define the short_name, units, fill_value, long_name, and precision from the product table
    results[[i + 1]] <- ncdf4::ncvar_def(currentLayer$short_name, currentLayer$units, list(dimx, dimy), NULL, currentLayer$long_name, prec = precision, compression = 2)
  }

  # Now create the netCDF file with the defined variables
  if (file.exists(outFile)) {
    file.remove(outFile)
    print(paste("removed:", outFile))
  }
  ncout <- ncdf4::nc_create(outFile, results, force_v4 = TRUE)
  print(paste("created:", outFile))

  print(paste("[", currentYear, "]", "adding data layers"))
  # Now loop through the layers again, this time actually writing the image data to the file
  for (currentLayerIdx in 1:dim(productTable)[1]) {
    currentLayer <- productTable[currentLayerIdx, ]

    # Open the data
    mat <- matrix(values(raster::raster(files[currentLayerIdx], varname = currentLayer$short_name)), length(x), length(y))
    # Assign fill value for pixels having values that outside of valid range or no-data
    mat[mat < currentLayer$valid_min | mat > currentLayer$valid_max | is.na(mat)] <- currentLayer$fill_value
    # Put the image into the file
    ncdf4::ncvar_put(ncout, results[[currentLayerIdx + 1]], mat)

    # Fill in the attributes for the layer from the product table
    ncdf4::ncatt_put(ncout, currentLayer$short_name, "scale", currentLayer$scale)
    ncdf4::ncatt_put(ncout, currentLayer$short_name, "offset", currentLayer$offset)
    ncdf4::ncatt_put(ncout, currentLayer$short_name, "data_type", currentLayer$data_type)
    ncdf4::ncatt_put(ncout, currentLayer$short_name, "valid_min", currentLayer$valid_min)
    ncdf4::ncatt_put(ncout, currentLayer$short_name, "valid_max", currentLayer$valid_max)

    print(paste(currentYear, "layer:", currentLayerIdx))
  }

  ## Write the projection info for the transverse_mercator variable
  # Get projection in wkt format
  # wkt <- showWKT(projection(baseImage), morphToESRI = FALSE)
  wkt <- sf::st_crs(baseImage)$wkt
  
  # Need to pull the central meridian from the wkt
  spt <- unlist(strsplit(gsub("]", "", wkt), ","))
  central_meridian <- as.numeric(spt[which(spt == "PARAMETER[\"central_meridian\"") + 1])

  print(paste("[", currentYear, "]", "filling nc info"))
  
  # Fill in the info.
  ncdf4::ncatt_put(ncout, "transverse_mercator", "long_name", "CRS definition")
  ncdf4::ncatt_put(ncout, "transverse_mercator", "grid_mapping_name", "transverse_mercator")
  ncdf4::ncatt_put(ncout, "transverse_mercator", "longitude_of_central_meridian", central_meridian)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "false_easting", 5e+05)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "false_northing", 0)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "latitude_of_projection_origin", 0)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "scale_factor_at_central_meridian", 0.9996)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "longitude_of_prime_meridian", 0)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "semi_major_axis", 6378137)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "inverse_flattening", 298.257223563)
  ncdf4::ncatt_put(ncout, "transverse_mercator", "GeoTransform", paste(base_extent[1], res, 0, base_extent[4], 0, -res))
  ncdf4::ncatt_put(ncout, "transverse_mercator", "spatial_ref", gsub("\\", "", wkt, fixed = TRUE))

  ## Define global attributes
  ncdf4::ncatt_put(ncout, 0, "title", "Land Surface Phenology from PlanetScope (PLSP)")
  ncdf4::ncatt_put(ncout, 0, "product_version", "v001")
  ncdf4::ncatt_put(ncout, 0, "summary", "A High Spatial Resolution Land Surface Phenology from PlancetScope for AmeriFlux and NEON sites")
  ncdf4::ncatt_put(ncout, 0, "software_repository", "git@github.com:BU-LCSC/PLSP.git")

  ncdf4::ncatt_put(ncout, 0, "creator_name", "Land Cover & Surface Climate Group, Department of Earth & Environment, Boston University")
  ncdf4::ncatt_put(ncout, 0, "creator_type", "group")
  ncdf4::ncatt_put(ncout, 0, "creator_email", "mkmoon@bu.edu")
  ncdf4::ncatt_put(ncout, 0, "creator_institution", "Boston University")

  ncdf4::ncatt_put(ncout, 0, "contributor_name", "Minkyu Moon, Andrew R. Richardson, Thomas Milliman, Mark A. Friedl")
  ncdf4::ncatt_put(ncout, 0, "contributor_role", "Developer, Co-Investigator, Collabolator, Principal Investigator")
  ncdf4::ncatt_put(ncout, 0, "acknowledgement", "This work was supported by NASA grant #80NSSC18K0334 and by NSF award #1702627.")

  # Put additional attributes on coordinates
  ncdf4::ncatt_put(ncout, "x", "axis", "projection_x_coordinate")
  ncdf4::ncatt_put(ncout, "y", "axis", "projection_y_coordinate")

  ## Close the file
  ncdf4::nc_close(ncout)
  
  print(paste("[", currentYear, "]", "closed"))
}

# summary stats
ncPhenoFiles <- list.files(path = paste0(outBase, strSite, "/"), pattern = glob2rx(paste0("*.nc")))
print(paste("# phenoYears:", length(phenoYears)))
print(paste("# phenoFiles:", length(ncPhenoFiles)))

print(paste("phenoYears:", paste(phenoYears, collapse = ", ")))
print(paste("phenoFiles:", paste(ncPhenoFiles, collapse = ", ")))
