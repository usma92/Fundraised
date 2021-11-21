# Load libraries
library(tidyverse)
library(skimr) # for describing data
library(Hmisc) # for describing data
library(corrplot) # for making correlation plot
library(caret) # for training
library(gbm) # for boosted regression
library(glmnet) # for LASSO

# Load and Describe Data -------------------------------------------------------
# Import data
setwd("~/GitHub/Fundraised")
csvName <- "InvidualParticipants.csv"
AllWalkData <- read.csv(csvName, stringsAsFactors=TRUE)

# Convert the integer variables to numerics
AllWalkData <- AllWalkData %>% mutate_if(is.integer, as.numeric)

summary(AllWalkData)
str(AllWalkData)
dim(AllWalkData)
AllWalkData %>% skim()
Hmisc::describe(AllWalkData)

# STEP 1 - DATA PREP, SPLIT 80/20 ----------------------------------------------

# Select variables to use in exploration models
AllWalkData.ModelVars <- AllWalkData %>%
  select(c(MeanLeadTime, MeanGoal, DistanceTraveled, SelfDonated, Fundraised,
           ModifiedGoal, UpdatedPersonalPage, SentEmails,
           CompletedReg, ProvidedParticipationReason, MeanTeamGoal,
           RepeatParticipant))

 # make index to split dataset
index.ModelVars <- createDataPartition(AllWalkData.ModelVars$SelfDonated,
                                       p=0.8, list=FALSE)

# Split dataset into train and test
df.train <- AllWalkData.ModelVars[index.ModelVars, ]
df.test <- AllWalkData.ModelVars[-index.ModelVars,  ]

# make transformed train dataset that is centered and scaled
df.train.preprocess <- df.train %>% preProcess(method = c("center", "scale"))
df.train.transformed <- df.train.preprocess %>% predict(df.train)

# STEP 2 - MODELING CRAZINESS --------------------------------------------------

# prepare training scheme
trCtrl <- trainControl(method="repeatedcv",
                       number=10,
                       repeats=3,
                       verboseIter=TRUE)

# LVQ --------------------------------------------------------------------------
# For LVQ, target variable must be a factor

# Make a new data frame AllWalkData.SelfDonated.LVQ to convert SelfDonated to a
# factor since LVQ needs the target variable as a factor


# train the model
model.SelfDonated.lvq <- train(SelfDonated~.,
                               data=df.train,
                               method="lvq",
                               preProcess=c("center","scale"),
                               trControl=trCtrl,
                               tuneLength=5)

# estimate variable importance
importance.SelfDonated.lvq <- varImp(model.SelfDonated.lvq, scale=FALSE)

# summarize importance
importance.SelfDonated.lvq
