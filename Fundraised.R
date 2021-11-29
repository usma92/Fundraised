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
library(rpart)
library(party)
library(partykit)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(gmodels)

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
#Use this for importance and Modeling choice
model.Fundraised.gbm = gbm(Fundraised ~ .,
                            data = df.train.transformed,
                            distribution = "gaussian",
                            n.trees = 5000,
                            shrinkage = 0.01,
                            interaction.depth = 4,
                            verbose = FALSE)

# Variable Importance
Fundraised.gbm.summary <- summary(model.Fundraised.gbm)
data.frame(var=Fundraised.gbm.summary$var, rel.inf=Fundraised.gbm.summary$rel.inf)





# STEP 3 - SELECT MOST IMPORTANT VARS, RUN DATA REDUCTION ----------------------

# Select Most Important Variables ----------------------------------------------
MostImportantVars.Fundraised <- df.train %>%
  select(c(MeanLeadTime, DistanceTraveled, SentEmails, MeanGoal, MeanTeamGoal,
           UpdatedPersonalPage, SelfDonated, ProvidedParticipationReason,
           RepeatParticipant, ModifiedGoal, CompletedReg, Fundraised))

MostImportantVars.Fundraised.test <- df.test %>%
  select(c(MeanLeadTime, DistanceTraveled, SentEmails, MeanGoal, MeanTeamGoal,
           UpdatedPersonalPage, SelfDonated, ProvidedParticipationReason,
           RepeatParticipant, ModifiedGoal, CompletedReg, Fundraised))

MostImportantVars.Fundraised.train <- df.train %>%
  select(c(MeanLeadTime, DistanceTraveled, SentEmails, MeanGoal, MeanTeamGoal,
           UpdatedPersonalPage, SelfDonated, ProvidedParticipationReason,
           RepeatParticipant, ModifiedGoal, CompletedReg, Fundraised))

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
                          NB=fit.nb, SVM=fit.svm, LVQ=fit.lvq))
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

#GLM Caret
glmGrid = expand.grid(.parameter = seq(1,10,1))
set.seed(7)
fit.glm <- train(Fundraised~., data=MostImportantVars.Model.Fundraised, method='glm', metric="Accuracy", 
                 trControl=trCtrl, tuneGrid = data.frame(glmGrid))
summary(fit.glm)

#attempt to use folding techniques
fit3.glm <- train(Fundraised~., data=MostImportantVars.Model.Fundraised,
               trControl = trCtrl, metric="Accuracy",
               method="glm", family="binomial")

summary(fit3.glm)

#model
summary(model.Fundraised.logit)
anova(model.Fundraised.logit, test="Chisq")

#Logit train results
fitted.logit.train.results <- predict(model.Fundraised.logit,newdata=subset(df.train,
                          select=c(MeanLeadTime, DistanceTraveled, SentEmails, 
                          MeanGoal, MeanTeamGoal,UpdatedPersonalPage, SelfDonated, 
                          ProvidedParticipationReason,RepeatParticipant, ModifiedGoal, 
                          CompletedReg)),type='response')

predictions.train.logit <- data.frame(Fundraised = df.train$Fundraised,
                                Pred.Fundraised = fitted.logit.train.results)

predictions.train.logit$Fundraised<-ifelse(predictions.train.logit$Fundraised=="Yes",1,0)
predictions.train.logit$Pred.Fundraised <- ifelse(predictions.train.logit$Pred.Fundraised > 0.3,1,0)

#Convert to factors for Confusion Matrix
predictions.train.logit$Fundraised <- as.factor(predictions.train.logit$Fundraised)
predictions.train.logit$Pred.Fundraised <- as.factor(predictions.train.logit$Pred.Fundraised)

#Creating confusion matrix
train.logit.confusion <- confusionMatrix(data=predictions.train.logit$Pred.Fundraised, reference = predictions.train.logit$Fundraised)
train.logit.confusion

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
predictions.logit$Pred.Fundraised <- ifelse(predictions.logit$Pred.Fundraised > 0.25,1,0)

#Convert to factors for Confusion Matrix
predictions.logit$Fundraised <- as.factor(predictions.logit$Fundraised)
predictions.logit$Pred.Fundraised <- as.factor(predictions.logit$Pred.Fundraised)

#Creating confusion matrix
logit.confusion <- confusionMatrix(data=predictions.logit$Pred.Fundraised, reference = predictions.logit$Fundraised)
logit.confusion

#Display results 
write.csv(predictions.logit, "./submission_logit.csv", row.names=FALSE)


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

#New decision tree based on Josh------------------------------------------------
#LASSO
LassoGrid <- expand.grid(
  alpha = 1,
  lambda = seq(0,1, length.out=100))

fitcontrol <- trainControl(method = "cv",number = 5)

Lasso <- train(Fundraised ~ .*.,
               data=MostImportantVars.Model.Fundraised,
               method = "glmnet",
               family = "binomial",
               tuneGrid=LassoGrid,
               trControl = fitcontrol)
interactionLASSO<-Lasso

coefINTERACTIONlasso<-coef(Lasso$finalModel,Lasso$bestTune$lambda) 
Lasso
summary(Lasso)

#MItchell's Lasso---------------------------------------------------------------

# LASSO with GLMNET ------------------------------------------------------------
df.Fundraised.train.glmnet <- MostImportantVars.Fundraised.train
df.Fundraised.test.glmnet <- MostImportantVars.Fundraised.test

model.Fundraised.glmnet <- glmnet(
  x=dplyr::select(df.Fundraised.train.glmnet, -Fundraised),
  y=df.Fundraised.train.glmnet$Fundraised,
  alpha=1,
  family="binomial",
  trace.it=1)

summary(model.Fundraised.glmnet)
coef(model.Fundraised.glmnet)
model.Fundraised.glmnet

#CV model run
model.Fundraised.CVglmnet <- cv.glmnet(
  x=data.matrix(dplyr::select(df.Fundraised.train.glmnet, -Fundraised)),
  y=df.Fundraised.train.glmnet$Fundraised,
  alpha=1,
  family="binomial",
  trace.it=1)

summary(model.Fundraised.CVglmnet)
coef(model.Fundraised.CVglmnet)
model.Fundraised.CVglmnet

bestLambda.Fundraised <- model.Fundraised.CVglmnet$lambda.min
bestLambda.Fundraised

#model run with best lambda from cv
model.Fundraised.bestCVglmnet <- glmnet(
  x=dplyr::select(df.Fundraised.train.glmnet, -Fundraised),
  y=df.Fundraised.train.glmnet$Fundraised,
  family="binomial",
  lambda=bestLambda.Fundraised,
  trace.it=1)

summary(model.Fundraised.bestCVglmnet)
coef(model.Fundraised.bestCVglmnet)
model.Fundraised.bestCVglmnet

# CV GLMnet Testing with best lambda model
# Testing across multiple thresholds, probably don't neet to use
# but just in case you want to speed up testing multiple thresholds
bestThreshAccuracy <- 0
bestAccuracy <- 0
bestThreshKappa <- 0
bestKappa <- 0  
for (predThresh in seq(0.3,0.7,by=0.01)) {
  print(predThresh)
  preds.Fundraised.CVglmnet <- predict(model.Fundraised.bestCVglmnet, 
                                        newx=data.matrix(dplyr::select(df.Fundraised.test.glmnet, -Fundraised)),
                                        s=bestLambda.Fundraised,
                                        type="response")
  preds.Fundraised.CVglmnet <- as.factor(c(ifelse(preds.Fundraised.CVglmnet > predThresh, 1, 0)))
  levels(preds.Fundraised.CVglmnet) <- c("No", "Yes")
  cm <- confusionMatrix(preds.Fundraised.CVglmnet, df.Fundraised.test.glmnet$Fundraised)
  if (cm$overall[1] > bestAccuracy) {
    bestThreshAccuracy <- predThresh
    bestAccuracy <- cm$overall[1]
  }
  
  if (cm$overall[2] > bestKappa) {
    bestThreshKappa <- predThresh
    bestKappa <- cm$overall[2]
  }
}
# Best Accuracy
# Best Kappa
# Testing across one threshold
preds.Fundraised.CVglmnet <- predict(model.Fundraised.bestCVglmnet, 
                                      newx=data.matrix(dplyr::select(df.Fundraised.test.glmnet, -Fundraised)),
                                      s=bestLambda.Fundraised,
                                      type="response")
preds.Fundraised.CVglmnet <- as.factor(c(ifelse(preds.Fundraised.CVglmnet > 0.7, 1, 0)))
levels(preds.Fundraised.CVglmnet) <- c("No", "Yes")
confusionMatrix(preds.Fundraised.CVglmnet, df.Fundraised.test.glmnet$Fundraised)

#Decisions tree one
DT1.Fundraised<-rpart(Fundraised~.,data=AllWalkData.ModelVars)

summary(DT1.Fundraised)
printcp(DT1.Fundraised)
plotcp(DT1.Fundraised)

DT1.Fundraised$cptable[which.min(DT1$cptable[,"xerror"]),"CP"]

#CP = 0.01

#Log Regression Summary based on notes
CrossTable(predictions.logit$Pred.Fundraised,predictions.logit$Fundraised, chisq=T)

#Original glm model
#fit.glm <- train(Fundraised~., data=df.train, method='glm', metric="Accuracy", 
#trControl=trCtrl)

fit2.glm <- glm(data=df.test,Fundraised~., family='binomial')
pred <- prediction(fit2.glm$fitted.values, fit2.glm$y)    #ROC curve for training data
perf <- performance(pred,"tpr","fpr") 

plot(perf,colorize=TRUE, print.cutoffs.at = c(0.25,0.5,0.75)); 
abline(0, 1, col="red")  


# can also plot accuracy by average cutoff level 
perf <- performance(pred, "acc")
plot(perf, avg= "vertical",  
     spread.estimate="boxplot", 
     show.spread.at= seq(0.1, 0.9, by=0.1))

# can also look at cutoff based on different cost structure
perf <- performance(pred, "cost", cost.fp=1, cost.fn=5)
plot(perf); 



# one nice one to look at...
# difference in distribution of predicted probabilities 
# for observations that were y=0 and y=1

plot(0,0,type="n", xlim= c(0,1), ylim=c(0,7),     
     xlab="Prediction", ylab="Density",  
     main="How well do the predictions separate the classes?")

for (runi in 1:length(pred@predictions)) {
  lines(density(pred@predictions[[runi]][pred@labels[[runi]]==1]), col= "blue")
  lines(density(pred@predictions[[runi]][pred@labels[[runi]]==0]), col="green")
}
#K-S chart  (Kolmogorov-Smirnov chart) 
# measures the degree of separation 
# between the positive (y=1) and negative (y=0) distributions

predVals$group<-cut(predVals$predProb,seq(1,0,-.1),include.lowest=T)
xtab<-table(predVals$group,predVals$trueVal)

xtab

#make empty dataframe
KS<-data.frame(Group=numeric(10),
               CumPct0=numeric(10),
               CumPct1=numeric(10),
               Dif=numeric(10))

#fill data frame with information: Group ID, 
#Cumulative % of 0's, of 1's and Difference
for (i in 1:10) {
  KS$Group[i]<-i
  KS$CumPct0[i] <- sum(xtab[1:i,1]) / sum(xtab[,1])
  KS$CumPct1[i] <- sum(xtab[1:i,2]) / sum(xtab[,2])
  KS$Dif[i]<-abs(KS$CumPct0[i]-KS$CumPct1[i])
}

KS  

KS[KS$Dif==max(KS$Dif),]

maxGroup<-KS[KS$Dif==max(KS$Dif),][1,1]

#and the K-S chart
ggplot(data=KS)+
  geom_line(aes(Group,CumPct0),color="blue")+
  geom_line(aes(Group,CumPct1),color="red")+
  geom_segment(x=maxGroup,xend=maxGroup,
               y=KS$CumPct0[maxGroup],yend=KS$CumPct1[maxGroup])+
  labs(title = "K-S Chart", x= "Deciles", y = "Cumulative Percent")

#Predictions--------------------------------------------------------------------

# Decision Tree
preds.dt <- predict(DT1.Fundraised, MostImportantVars.Fundraised.test,
                    type="class")

confusionMatrix(preds.dt, MostImportantVars.Fundraised.test$Fundraised)
DT1.Fundraised$cptable[which.min(DT1$cptable[,"xerror"]),"CP"] #prints out the min CP value

# Logistic (GLM)
preds.glm <- predict(model.Fundraised.glm, df.Fundraised.test.glm,
                     type="response")
preds.glm <- as.factor(c(ifelse(preds.glm > 0.5, 1, 0)))
levels(preds.glm) <- c("No", "Yes")
confusionMatrix(preds.glm, df.Fundraised.test.glm$Fundraised)

# Lasso with Interactions
preds.lassoInteract <- predict(bestmodel.Fundraised.CVglmnet, 
                               MostImportantVars.Fundraised.test)
confusionMatrix(preds.lassoInteract, Fundraised.test.$Fundraised)