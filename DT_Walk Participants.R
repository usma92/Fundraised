library(tidyverse)
library(caret)
library(gbm)
library(glmnet)

#install.packages("gbm")


library(rpart)          # for decision tree modeling
library(party)          # for visualizing trees
library(partykit)       # for visualizing trees
library(ggplot2)        # for graphics
library(ROCR)           # for graphics
library(rattle)      		# fancy tree plot
library(rpart.plot)			# enhanced tree plots
library(RColorBrewer)

setwd("~/GitHub/Fundraised")
AllWalkData<-read_csv("All Registrants Combined 2017-2019 Cleaned_Scrubbed_St_Address_with_zip_distances.csv")

#clean up NA
#AllWalkData <-AllWalkData[AllWalkData =="" | AllWalkData==" "] 
#data.frame(colnames(AllWalkData))


############################
#####Creating Features#####
###########################

# Financially Contributed in Any Way

AllWalkData$FinanciallyContributed <- with(AllWalkData, ifelse(AllWalkData$`Revenue Raised` > 0, "Yes", "No"))

#Personally Donated
AllWalkData$SelfDonated <- with(AllWalkData, ifelse(AllWalkData$`Self Donation` > 0 | AllWalkData$`Total From Participant($)` > 0 , "Yes", "No"))

#Fundraised
AllWalkData$Fundraised <- with(AllWalkData, ifelse(AllWalkData$`Total Not From Participant($)` > 0, "Yes", "No"))

#Modified Goal
AllWalkData$ModifiedGoal <- with(AllWalkData, ifelse(!AllWalkData$Goal == 250, "Yes", "No"))

# On a team
AllWalkData$OnTeam <- with(AllWalkData, ifelse(!is.na(AllWalkData$`Team Name`) > 0, "Yes", "No"))

#Updated Personal Page
AllWalkData$UpdatedPersonalPage <- with(AllWalkData, ifelse(!is.na(AllWalkData$`Personal Page Last Updated Date`) > 0, "Yes", "No"))

#Sent Fundraising Emails
AllWalkData$SentEmails <- with(AllWalkData, ifelse(AllWalkData$`Emails Sent` > 0, "Yes", "No"))

#Completed profile Registration

AllWalkData$FullyCompletedRegistration<-with(AllWalkData, ifelse(!is.na(AllWalkData$`Address 1`) & !is.na(AllWalkData$City) &
                           !is.na(AllWalkData$State) & !is.na(AllWalkData$Zip) &
                           !is.na(AllWalkData$Email) & !is.na(AllWalkData$Employer)&
                           !is.na(AllWalkData$`Participant Gender`) & !is.na(AllWalkData$`Participant I am walking`) 
                           > 0, "Yes", "No"))

#Provided Participation Reason
AllWalkData$ProvidedParticipationReason<- with(AllWalkData, ifelse(!is.na(AllWalkData$`Participant I am walking`) > 0, "Yes", "No"))

#Repeat Participant

# Find where the Name value occurs more than 1 time (meaing the person has participated in more than 1 walk)
NameWalksGreaterThan1 <- pull(AllWalkData %>% group_by(Name) %>% filter(n() > 1) %>% select(Name), Name)
# Set the default value for RepeatParticipants to "No"
AllWalkData$RepeatParticipants <- rep_len("No", length(AllWalkData$Name))

# For each value in NameWalksGreaterThan1
for (i in 1:length(NameWalksGreaterThan1)) {
  # Find all row numbers in AllWalkData where the name of the repeat participant's name occurs
  rowNums <- which(AllWalkData$Name == NameWalksGreaterThan1[i], arr.ind=TRUE)
  # for each row number found, set the value of AllWalkData$RepeatParticipants to "Yes"
  for (j in 1:length(rowNums)) {
    AllWalkData$RepeatParticipants[rowNums[j]] = "Yes"
  }
}

##converting distances to buckets (Not working yet--might be better to do on the averaging)
AllWalkData %>% 
  mutate(DistanceBuckets = case_when(
    between(AllWalkData$distance_miles, 0, 50) ~ "[0-50]",
    between(AllWalkData$distance_miles, 51, 100) ~ "[51-100]",
    between(AllWalkData$distance_miles, 101, 150) ~ "[101-150]",
    between(AllWalkData$distance_miles, 151, 2500) ~ "[151-2500]",
    TRUE ~ NA_character_
  ))


####FCT Collapse on Team Division####


######Help here with NA Values##########
AllWalkData %>% 
  group_by(`Team Division`) %>% 
  summarise(n=n())

AllWalkData$`Team Division`<-fct_other(
  AllWalkData$`Team Division`, keep = c("Community Group", "Corporate", "Dialysis","Family/Friends","School"))

AllWalkData$`Team Division`<-fct_collapse(AllWalkData$`Team Division`,
                                          CommunityGroup = c("Community Group","School"),
                                          Corporate = "Corporate",
                                          Dialysis = "Dialysis",
                                          FamilyFriends = "Family/Friends",
                                          None = c("","Other"))

# Make Gender NA values a level of "UNKNOWN"
AllWalkData$`Participant Gender` <- fct_explicit_na(AllWalkData$`Participant Gender`, "UNKNOWN")

###########################
########Event Grouping#####
###########################

dfwalk_group <- AllWalkData %>% 
  group_by(`Walk Name...1`) %>% 
  summarise( n = n(),
             MeanLeadTime = mean(RegistrationLeadTime),
             MeanGoal = mean(Goal),
             TotalRevenue = sum(`Revenue Raised`),
             Revenue_Participant = TotalRevenue/n, 
             TotalFundraising = sum(`Total Not From Participant($)`),
             Fundraising_Participant = TotalFundraising/n,
             TotalSelfDonation = sum(`Revenue Raised`)-sum(`Total Not From Participant($)`),
             SelfDonation_Participant = TotalSelfDonation/n,
             P_SelfDonation = TotalSelfDonation/TotalRevenue,
             P_Fundraising = TotalFundraising/TotalRevenue, 
             Employer=names(which(table(Employer) == max(table(Employer)))[1]),
             Gender=names(which(table(`Participant Gender`) == max(table(`Participant Gender`)))[1]),
             Avg_Email = mean(`Emails Sent`),
             Mean_Distance = mean(ifelse(is.na(distance_miles),0,distance_miles)),
             n_FinanciallyContributed = sum(`FinanciallyContributed` == "Yes"),
             P_FinanciallyContributed = n_FinanciallyContributed/n,
             n_SelfDonated = sum(SelfDonated == "Yes"),
             P_Part_SelfDonated = n_SelfDonated/n,
             n_Part_Fundraised= sum(Fundraised == "Yes"),
             P_Part_Fundraised = n_Part_Fundraised/n,
             n_ModifiedGoal = sum(ModifiedGoal == "Yes"),
             P_ModifiedGoal = n_ModifiedGoal/n,
             n_OnTeam = sum(OnTeam == "Yes"),
             P_OnTeam = n_OnTeam/n,
             n_CompletedRegistration = sum(FullyCompletedRegistration == "Yes"),
             P_CompletedRegistration = n_CompletedRegistration/n,
             n_UpdatedPersonalPage = sum(UpdatedPersonalPage=="Yes"),
             P_UpdatedPersonalPage = n_UpdatedPersonalPage/n)

par(mfrow=c(3,2))
qqplot(dfwalk_group$Mean_Distance,dfwalk_group$Revenue_Participant)
#shows pretty linear relationship between distance and revenue per participant

qqplot(dfwalk_group$MeanLeadTime,dfwalk_group$Revenue_Participant)
#Again shows a pretty stong relationship (probably a good candidate for Mars modeling)

qqplot(dfwalk_group$P_ModifiedGoal,dfwalk_group$Revenue_Participant)
#yep similar pattern

qqplot(dfwalk_group$P_OnTeam,dfwalk_group$Revenue_Participant)
#pretty interesting linear relationship with a BIG jump at 0.8

qqplot(dfwalk_group$P_OnTeam,dfwalk_group$Mean_Distance)
#interesting mean distance and percentage on a team is releated

####################################
######Individual Grouping###########
####################################


dfwalk_ind <- AllWalkData %>%
  group_by(Name) %>%
  summarise(n=n(),
            MeanLeadTime = mean(RegistrationLeadTime),
            MeanGoal = mean(Goal),
            MeanRevenue = mean(ifelse(is.na(`Revenue Raised`),0,`Revenue Raised`)),
            MeanFundraising = mean(ifelse(is.na(`Total Not From Participant($)`),0,`Total Not From Participant($)`)),
            MeanSelfDonnation = MeanRevenue-MeanFundraising,
            DistanceTraveled = mean(ifelse(is.na(distance_miles),0,distance_miles)),
            #TeamDivision = names(which(table(`Team Division`) == max(table(`Team Division`)))[1]),
            FinanciallyCont = names(which(table(FinanciallyContributed) == max(table(FinanciallyContributed)))[1]), #either donnated or raised money
            SelfDonated=names(which(table(SelfDonated) == max(table(SelfDonated)))[1]),
            Fundraised = names(which(table(Fundraised) == max(table(Fundraised)))[1]),
            Gender=names(which(table(`Participant Gender`) == max(table(`Participant Gender`)))[1]),
            ModifiedGoal=names(which(table(ModifiedGoal) == max(table(ModifiedGoal)))[1]),
            Onteam = names(which(table(OnTeam) == max(table(OnTeam)))[1]),
            UpdatedPersonalPage = names(which(table(UpdatedPersonalPage) == max(table(UpdatedPersonalPage)))[1]),
            SentEmails = names(which(table(SentEmails) == max(table(SentEmails)))[1]),
            CompletedReg = names(which(table(FullyCompletedRegistration) == max(table(FullyCompletedRegistration)))[1]),
            ProvidedParticipationReason = names(which(table(ProvidedParticipationReason) == max(table(ProvidedParticipationReason)))[1]),
            MeanTeamGoal = mean(ifelse(is.na(`Team Goal($)`),0,`Team Goal($)`)),
            RepeatParticipant=names(which(table(RepeatParticipants) == max(table(RepeatParticipants)))[1]))

dfwalk_ind %>% 
  group_by(RepeatParticipant) %>% 
  summarise(n=n())


#dealing with the character not being factors
write.csv(dfwalk_ind,file = "InvidualParticipants.csv")
InvidualParticipants <-read.csv("InvidualParticipants.csv",stringsAsFactors=TRUE)
str(InvidualParticipants)

#make column names compatible with R rules
colnames(InvidualParticipants) <- make.names(colnames(InvidualParticipants))
      

#Creating Test and Train
train <- slice_sample(InvidualParticipants, prop = 0.8, replace = FALSE)
test <- slice_sample(InvidualParticipants, prop = 0.2, replace = FALSE)

#Select Variables to use in the model from Train
str(train)

df_train <- train %>% 
  select(c(MeanLeadTime,DistanceTraveled,MeanGoal,SelfDonated,Fundraised,ModifiedGoal,Onteam,
           UpdatedPersonalPage,SentEmails,CompletedReg,ProvidedParticipationReason,
           MeanTeamGoal,RepeatParticipant))

df1.preprocess <- df_train %>% preProcess(method = c("center", "scale"))
df1.transformed <- df1.preprocess %>% predict(df_train)

#boosted tree model
model.gbm = gbm(RepeatParticipant ~ ., data = df_train, distribution = "gaussian",
                n.trees = 1000, shrinkage = 0.01, interaction.depth = 4)

# Variable Importance Plot
summary(model.gbm)


#######################
#####LASSO#########
#################

#define response variable
y<-df_train$RepeatParticipant


#define matrix of predictor variables
x<-data.matrix(train[,c("MeanLeadTime","DistanceTraveled","MeanGoal","SelfDonated","Fundraised","ModifiedGoal","Onteam",
                        "CompletedReg","ProvidedParticipationReason","MeanTeamGoal")])
lasso<-glmnet(x=x, 
              y=y,
              family = binomial)


cv_lasso <- cv.glmnet(x=x, 
                   y=y,
                   alpha = 1,
                   family = binomial)

best_lamda <- cv_lasso$lambda.min

best_lamda


best_model <- glmnet(x, y, alpha = 1, 
                     family = binomial, lambda = best_lamda)
coef(best_model)

