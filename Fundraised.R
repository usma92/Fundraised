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
library(ROCR)
library(earth)

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
#write.csv(AllWalkData, "./AllWalkData.csv", row.names=FALSE)
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

# LVQ Classification--------------------------------------------------------------------------
# train the model
par(mfrow=c(1,1))
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

# Boosted Tree Classification-----------------------------------------------------------------
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
corrplot(correlationMatrix.Fundraised)

# Find the values in the correlation matrix that are highly correlated (value
# greater than 0.5)
highlyCorrelated.Fundraised <-
  findCorrelation(correlationMatrix.Fundraised, cutoff=0.5, names=TRUE)
print(highlyCorrelated.Fundraised)

#Compare base regression models-----------------------------------------------------------------
#GLM
set.seed(7)
fit.glm <- train(Fundraised~., data=df.train, method='glm', metric="Accuracy", 
                 trControl=trCtrl)

#LDA
set.seed(7)
fit.lda <- train(Fundraised~., data=df.train, method='lda', metric="Accuracy", 
                 trControl=trCtrl)

#GLMNET
set.seed(7)
fit.glmnet <- train(Fundraised~., data=df.train, method='glmnet', metric="Accuracy", 
                 trControl=trCtrl)

#KNN
set.seed(7)
fit.knn <- train(Fundraised~., data=df.train, method='knn', metric="Accuracy", 
                 trControl=trCtrl)

#Naive Bayes
set.seed(7)
fit.nb <- train(Fundraised~., data=df.train, method='nb', metric="Accuracy", 
                 trControl=trCtrl)

#SVM
set.seed(7)
fit.svm <- train(Fundraised~., data=df.train, method='svmRadial', metric="Accuracy", 
                 trControl=trCtrl)

#LVQ
set.seed(7)
fit.lvq <- train(Fundraised~., data=df.train, method='lvq', metric="Accuracy", 
                 trControl=trCtrl)

#Compare algorithms
results <- resamples(list(LG=fit.glm, LDA=fit.lda, GLMNET=fit.glmnet, KNN=fit.knn,
                          CART=fit.cart, NB=fit.nb, SVM=fit.svm, LVQ=fit.lvq))
summary(results)
dotplot(results)

#Base algorithms with transformed datasets -------------------------------------

#GLM
set.seed(7)
fit.glm <- train(Fundraised~., data=df.train.transformed, method='glm', metric="Accuracy", 
                 trControl=trCtrl)

#LDA
set.seed(7)
fit.lda <- train(Fundraised~., data=df.train.transformed, method='lda', metric="Accuracy", 
                 trControl=trCtrl)

#GLMNET
set.seed(7)
fit.glmnet <- train(Fundraised~., data=df.train.transformed, method='glmnet', metric="Accuracy", 
                    trControl=trCtrl)

#KNN
set.seed(7)
fit.knn <- train(Fundraised~., data=df.train.transformed, method='knn', metric="Accuracy", 
                 trControl=trCtrl)

#Naive Bayes
set.seed(7)
fit.nb <- train(Fundraised~., data=df.train.transformed, method='nb', metric="Accuracy", 
                trControl=trCtrl)

#SVM
set.seed(7)
fit.svm <- train(Fundraised~., data=df.train.transformed, method='svmRadial', metric="Accuracy", 
                 trControl=trCtrl)

#LVQ
set.seed(7)
fit.lvq <- train(Fundraised~., data=df.train, method='lvq', metric="Accuracy", 
                 trControl=trCtrl)

#Compare algorithms
results <- resamples(list(LG=fit.glm, LDA=fit.lda, GLMNET=fit.glmnet, KNN=fit.knn,
                          NB=fit.nb, SVM=fit.svm, LVQ=fit.lvq))
summary(results)
dotplot(results)

#Ensemble Methods
# Bagged CART
set.seed(7)
fit.treebag <- train(Fundraised~., data=df.train.transformed, method="treebag", metric="Accuracy",
                     trControl=trCtrl, na.action=na.omit)

# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(Fundraised~., data=df.train.transformed, method="gbm", metric="Accuracy",
                     trControl=trCtrl, na.action=na.omit)

# Random Forest
set.seed(7)
fit.rf <- train(Fundraised~., data=df.train.transformed, method="rf", metric="Accuracy",
                     trControl=trCtrl, na.action=na.omit)

# Compare results 
ensembleResults <- resamples(list(BAG=fit.treebag, RF=fit.rf, GBM=fit.gbm))
summary(ensembleResults)
dotplot(ensembleResults)
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

#Logit Regression---------------------------------------------------------------
MostImportantVars.Model.Fundraised <- df.train %>%
  select(c(MeanLeadTime, DistanceTraveled, SentEmails, MeanGoal, MeanTeamGoal,
           UpdatedPersonalPage, SelfDonated, ProvidedParticipationReason,
           RepeatParticipant, ModifiedGoal, CompletedReg,Fundraised))

model.Fundraised.logit <- glm(Fundraised ~MeanLeadTime+DistanceTraveled+SentEmails+
                               MeanGoal+MeanTeamGoal+UpdatedPersonalPage+SelfDonated+
                               ProvidedParticipationReason+RepeatParticipant+ 
                               ModifiedGoal+CompletedReg,
                               family=binomial(link='logit'),data=MostImportantVars.Model.Fundraised)

summary(model.Fundraised.logit)

anova(model.Fundraised.logit, test="Chisq")

#Predictions Logit Regression---------------------------------------------------
fitted.results <- predict(model.Fundraised.logit,newdata=subset(df.test,
                          select=c(MeanLeadTime, DistanceTraveled, SentEmails, 
                          MeanGoal, MeanTeamGoal,UpdatedPersonalPage, SelfDonated, 
                          ProvidedParticipationReason,RepeatParticipant, ModifiedGoal, 
                          CompletedReg)),type='response')


# Logit Predictions on hold out
predictions.logit <- data.frame(Fundraised = df.test$Fundraised,
                              Pred.Fundraised = fitted.results)

predictions.logit$Fundraised<-ifelse(predictions.logit$Fundraised=="Yes",1,0)
predictions.logit$Pred.Fundraised <- ifelse(predictions.logit$Pred.Fundraised > 0.3,1,0)

#Convert to factors for Confusion Matrix
predictions.logit$Fundraised <- as.factor(predictions.logit$Fundraised)
predictions.logit$Pred.Fundraised <- as.factor(predictions.logit$Pred.Fundraised)

#Creating confusion matrix
logit.confusion <- confusionMatrix(data=predictions.logit$Pred.Fundraised, reference = predictions.logit$Fundraised)
logit.confusion

#Display results 
write.csv(predictions.logit, "./submission_logit.csv", row.names=FALSE)

#MARS---------------------------------------------------------------------------
marsFormula <- Fundraised ~MeanLeadTime+DistanceTraveled+SentEmails+
  MeanGoal+MeanTeamGoal+UpdatedPersonalPage+SelfDonated+
  ProvidedParticipationReason+RepeatParticipant+ 
  ModifiedGoal+CompletedReg

# set the preferred degree and number of values to prune
tGrid <- expand.grid(degree=3, nprune=18)

model.mars <- train(marsFormula, data=df.train, method="earth", tuneGrid=tGrid)
summary(mars)
mars

#Predictions MARS Regression---------------------------------------------------
fit.mars<- predict(model.mars,newdata=subset(df.test,select=c(MeanLeadTime, 
                                            DistanceTraveled, SentEmails,MeanGoal, 
                                            MeanTeamGoal,UpdatedPersonalPage, 
                                            SelfDonated, ProvidedParticipationReason,
                                            RepeatParticipant, ModifiedGoal,CompletedReg)))


# Mars Predictions on hold out
predictions.mars <- data.frame(Fundraised = df.test$Fundraised,
                                Pred.Fundraised = fitted.results)
str(predictions.mars)


predictions.mars$Fundraised<-ifelse(predictions.mars$Fundraised=="Yes",1,0)
predictions.mars$Pred.Fundraised <- ifelse(predictions.mars$Pred.Fundraised > 0.4,1,0)

#Convert to factors for Confusion Matrix
predictions.mars$Fundraised <- as.factor(predictions.mars$Fundraised)
predictions.mars$Pred.Fundraised <- as.factor(predictions.mars$Pred.Fundraised)

#Creating confusion matrix
mars.confusion <- confusionMatrix(data=predictions.mars$Pred.Fundraised, reference = predictions.mars$Fundraised)
mars.confusion

#Display results 
write.csv(predictions.logit, "./submission_mars.csv", row.names=FALSE)
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
# Plot Importance of LVQ
png(filename = "FundraisedGraphs/LVQImportance.png")
plot(importance.Fundraised.lvq)
dev.off()

png(filename = "FundraisedGraphs/GBMVariableImportance.png")
#summary(gbm.fit)
print(gbm.summary)
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
