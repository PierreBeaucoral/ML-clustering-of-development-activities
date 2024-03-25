######### Packages ############

rm(list = ls())
library(readr)
library(png)
library(purrr)
library(gtable)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
library(pander)
library(tidyverse)
library(tmap)
library(leaflet)
library(ggforce)
library(treemap)
library(readxl)
library(plotly)
library(ggalluvial)
library(migest)
library(rlist)
library(data.table)
library(hms)
library(flextable)
library(skimr)


##### Déclarer le chemin des bases #####

# Define the desired working directory path
wd <- "/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/Travaux CRS"

# Get the current working directory
current_wd <- getwd()

# Check if the current working directory is different from the desired directory
if (current_wd != wd) {
  # If different, set the working directory to the desired directory
  setwd(wd)
  print(paste("Changed working directory to:", getwd()))
} else {
  # If already in the desired directory, print a message indicating the current directory
  print(paste("Current working directory is already set to:", getwd()))
}

####### Specification des années prises en compte #####

##### Années de début/fin ####

Period <- c(1973:2022) 
Bound  <- paste0(min(Period), "-", max(Period)) 

### Base avec des données annuelles ###

Yearly <- c(2006: 2021)

######### Chargement des bases ########

##### Bases annuelles #####

for (i in  Yearly) {
  filename<-paste0("CRS"," ",i," ", "Data")
  wd<-paste0("./Data/CRS/CRS"," ",i," ", "Data", ".txt")
  assign(filename,fread(wd,  encoding =  "Latin-1"))
}

gc() 

#### Création d'une liste des bases ####

CRS <- lapply(Yearly, function(x) {
  dataframeName <- paste0("CRS ", x, " Data")
  return(get(dataframeName))
})

##### Bases pluri-annuelles #####


"CRS 1973-94 data" <- fread("./Data/CRS/CRS 1973-94 data.txt",  encoding =  "Latin-1")
"CRS 1995-99 data" <- fread("./Data/CRS/CRS 1995-99 data.txt",  encoding =  "Latin-1")
"CRS 2000-01 data" <- fread("./Data/CRS/CRS 2000-01 data.txt",  encoding =  "Latin-1")
"CRS 2002-03 data" <- fread("./Data/CRS/CRS 2002-03 data.txt",  encoding =  "Latin-1")
"CRS 2004-05 data" <- fread("./Data/CRS/CRS 2004-05 data.txt",  encoding =  "Latin-1")


####### Création d'une liste avec toutes les données #######

#### Création de la liste ####
BDD <- list()
gc() #libérer de l'espace#

#### Insertion des bases ####

BDD <-  c(list(`CRS 1973-94 data`, `CRS 1995-99 data`, `CRS 2000-01 data`, `CRS 2002-03 data`, `CRS 2004-05 data`),CRS)
gc() #libérer de l'espace#

#### Changer les éléments de la liste en DataFrame #### 

BDD <- lapply(1:length(BDD), function (x){
  as.data.frame(BDD[x])
})
gc()


out <- lapply(1:length(BDD), function (x){
  split( BDD[[x]] , f = BDD[[x]]$Year )
  }) 
gc()



BDD <- unlist(out, recursive = FALSE)
gc()

rm(list=setdiff(ls(), c("BDD", "CRS", "Period", "Bound")))
gc()
