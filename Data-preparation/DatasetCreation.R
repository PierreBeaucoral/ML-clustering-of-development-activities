rm(list = ls())

# Define the desired working directory path
wd <- "/Your/Path"

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

#### 0- Import data from .txt files#### 

source("UploadBase.R")
gc()


#### 1- Get only data of interest #### 

BDD <- lapply (BDD, function(x) drop_na(x, c("Year", "DonorCode", "RecipientCode")))
rm(list=setdiff(ls(), c("BDD", "Bound", "Period")))
gc()

#### 2- Create variable raw_text #### 

processed_df_list <- lapply(BDD, function(df) {
  df <- df %>%
    select(Year, DonorCode, DonorName, CrsID, RecipientCode, RecipientName, FlowCode, Bi_Multi, Category, Finance_t, USD_Commitment_Defl, USD_Disbursement_Defl, ProjectTitle, ShortDescription, LongDescription, PurposeCode) %>%
    filter(!is.na(ProjectTitle) & !is.na(ShortDescription) & !is.na(LongDescription)) %>%
    unite("raw_text", ProjectTitle:ShortDescription, sep = ' ') %>%
    unite("raw_text", raw_text:LongDescription, sep = ' ')
  gc()
  return(df)
})
gc()


### Lowercasing ###

processed_df_list <- lapply(processed_df_list, function(df) {
  df$raw_text <- tolower(df$raw_text)
  return(df)
})

#### 3- Create  Dataframe #### 

Data <- do.call("rbind", processed_df_list)
gc()

### remove duplicates ### 
               
Data <- Data%>%
  distinct()
gc() 

#### 4- Remove project descriptions where there are only "semi-aggregates" and "sectors not specified" expressions #### 

Data <- Data%>%
  filter(!grepl("semi-aggregates", raw_text, fixed = TRUE))%>%
  filter(!grepl("sectors not specified", raw_text, fixed = TRUE))

# Remove underscores from the raw text variable
Data$raw_text <- gsub("_", "", Data$raw_text)

# Create a subsample for beta-testing
rand_df <- Data[sample(nrow(Data), size=10000), ]

write.table(Data, "./Data/Dataset.csv", sep="|")
write.table(rand_df, "./Data/Datasample.csv", sep="|")
gc()
