# Load libraries
library(tidyverse)
library(skimr) # for describing data
library(Hmisc) # for describing data
library(corrplot) # for making correlation plot
library(caret) # for training
library(gbm) # for boosted regression
library(glmnet) # for LASSO
library(png)
library(corrplot)

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
# train the model
model.Fundraised.lvq <- train(Fundraised~.,
                               data=df.train,
                               method="lvq",
                               preProcess=c("center","scale"),
                               trControl=trCtrl,
                               tuneLength=5)

# estimate variable importance
importance.Fundraised.lvq <- varImp(model.Fundraised.lvq, scale=FALSE)

# summarize importance
importance.Fundraised.lvq

#LASSO -------------------------------------------------------------------------

#define response variable
y<-df.train$Fundraised

#define matrix of predictor variables
x<-data.matrix(subset(df.train, select = -c(Fundraised)))

model.Fundraised.cv_lasso <- cv.glmnet(x=x, y=y,alpha = 1,family = binomial)
best_lamda <- model.Fundraised.cv_lasso$lambda.min
best_lamda
best_model <- glmnet(x, y, alpha = 1, 
                     family = binomial, lambda = best_lamda)
coef(best_model)

# Boosted Tree -----------------------------------------------------------------
model.Fundraised.gbm = gbm(Fundraised ~ .,
                            data = df.train,
                            distribution = "gaussian",
                            n.trees = 5000,
                            shrinkage = 0.01,
                            interaction.depth = 4,
                            verbose = TRUE)

# Variable Importance
Fundraised.gbm.summary <- summary(model.Fundraised.gbm)
data.frame(var=Fundraised.gbm.summary$var, rel.inf=Fundraised.gbm.summary$rel.inf)
plot(model.Fundraised.gbm, i = "MeanLeadTime")
# STEP 3 - SELECT MOST IMPORTANT VARS, RUN DATA REDUCTION ----------------------

# Select Most Important Variables ----------------------------------------------
MostImportantVars.Fundraised <- df.train %>%
  select(c(MeanLeadTime, DistanceTraveled, SentEmails, MeanGoal, MeanTeamGoal,
           UpdatedPersonalPage, SelfDonated, ProvidedParticipationReason,
           RepeatParticipant, ModifiedGoal, CompletedReg))

MostImportantVars.Fundraised.Integer <- data.frame(lapply(MostImportantVars.Fundraised, as.integer))

# Correlation Matrix -----------------------------------------------------------
# Make a correlation matrix for the most important vars
correlationMatrix.Fundraised <- cor(MostImportantVars.Fundraised.Integer)
print(correlationMatrix.Fundraised)

# Find the values in the correlation matrix that are highly correlated (value
# greater than 0.5)
highlyCorrelated.Fundraised <-
  findCorrelation(correlationMatrix.Fundraised, cutoff=0.5, names=TRUE)
print(highlyCorrelated.Fundraised)

# Important Plots of Analysis --------------------------------------------------

# Plot Importance of LVQ
png(filename = "FundraisedGraphs/LVQImportance.png")
plot(importance.Fundraised.lvq)
dev.off()

# Correlation plot
png(filename = "FundraisedGraphs/Correlation.png")
corrplot(correlationMatrix.Fundraised,
         method='ellipse',
         type='lower',
         tl.col='black',
         tl.srt = 45,
         tl.cex=0.6)
dev.off()

# Variable Importance Plot of GBM
png(filename = "FundraisedGraphs/GBMVariableImportance.png")
#summary(model.Fundraised.gbm)
Fundraised.gbm.summary
dev.off()

# Partial Dependence Plots from GBM
png(filename = "FundraisedGraphs/MeanLeadTimePartialDependence.png")
plot(model.Fundraised.gbm, i = "MeanLeadTime")
dev.off()

png(filename = "FundraisedGraphs/DistanceTraveledPartialDependence.png")
plot(model.Fundraised.gbm, i = "DistanceTraveled")
dev.off()

png(filename = "FundraisedGraphs/SentEmailsPartialDependence.png")
plot(model.Fundraised.gbm, i = "SentEmails")
dev.off()

png(filename = "FundraisedGraphs/MeanGoalPartialDependence.png")
plot(model.Fundraised.gbm, i = "MeanGoal")
dev.off()

png(filename = "FundraisedGraphs/MeanTeamGoalPartialDependence.png")
plot(model.Fundraised.gbm, i = "MeanTeamGoal")
dev.off()

png(filename = "FundraisedGraphs/UpdatedPersonalPagePartialDependence.png")
plot(model.Fundraised.gbm, i = "UpdatedPersonalPage")
dev.off()

png(filename = "FundraisedGraphs/SelfDonatedPartialDependence.png")
plot(model.Fundraised.gbm, i = "SelfDonated")
dev.off()

png(filename = "FundraisedGraphs/ModifiedGoalPartialDependence.png")
plot(model.Fundraised.gbm, i = "ModifiedGoal")
dev.off()

png(filename = "FundraisedGraphs/ProvidedParticipationReasoPartialDependence.png")
plot(model.Fundraised.gbm, i = "ProvidedParticipationReason")
dev.off()

png(filename = "FundraisedGraphs/RepeatParticipantPartialDependence.png")
plot(model.Fundraised.gbm, i = "RepeatParticipant")
dev.off()

png(filename = "FundraisedGraphs/CompletedRegPartialDependence.png")
plot(model.Fundraised.gbm, i = "CompletedReg")
dev.off()

# GBM Performance Plot
numTrees = seq(from=100 ,to=5000, by=100)

df.test$Fundraised <- as.numeric(df.test$Fundraised)
predmatrix<-predict(model.Fundraised.gbm, df.test, n.trees = numTrees)
dim(predmatrix)

test.error<-with(df.test,apply( (predmatrix-Fundraised)^2,2,mean))
head(test.error)

png(filename = "FundraisedGraphs/GBMPerformance.png")
plot(numTrees,
     test.error, 
     pch=19,
     col="blue",
     xlab="Number of Trees",
     ylab="Test Error", 
     main = "Perfomance of Boosting on Test Set")

abline(h = min(test.error),col="red")
legend("topright",c("Minimum Test error Line for Random Forests"),col="red",lty=1,lwd=1)
dev.off()

# Plot of CV Lasso Model
png(filename = "FundraisedGraphs/LassoWithCV.png")
plot(model.Fundraised.cv_lasso)
dev.off()
