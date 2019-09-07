### Regression Kriging example code
### Dan Schmutz
### 20190904
### Prepared for the Orlando Machine Learning and Data Science Meetup on 9/7/2019

# Packages to load
library(visreg)
library(glmulti)
library(car)
library(caret)
library(dplyr)
library(rattle)
library(QuantPsyc)
library(automap)
library(GSIF)
library(randomForest)
library(tidyr)
library(readr)

# Importing data
wetlakenoaugonly2008 <- read_csv("wetlakenoaugonly2008.csv")

data<-wetlakenoaugonly2008 
data$postcutddn[data$postcutddn < -98]<-NA # filling some missing data with NAs
write.csv(data,file='datastart.csv',row.names=F)

# Now Selecting 80% of data as sample from total 'n' rows of the data  
tempdata<-data
set.seed(42)
sample <- sample.int(n = nrow(tempdata), size = floor(0.80*nrow(tempdata)), replace = F)
train <- tempdata[sample, ]
test  <- tempdata[-sample, ]

# glmulti for finding BIC-best linear regression
glm1<-glmulti(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,data=train,crit="bic",level=1,method='d') # to check number of models
glm1<-glmulti(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,data=train,crit="bic",level=1,method='h') #  to run exhaustive fit of all possible models

plot(glm1,type="r") # plot of residuals of top 5 models
plot(glm1,type="s") # plot of variable importances
plot(glm1,type="w") # plot of model weights

lm2<-lm(npo2008trans ~ TBWDDN + RAXericRat,data=train) # linear regression on training data using BIC-best model
summary(lm2)
visreg(lm2) # prepare partial residual plots
vif(lm2) # checking variance inflation factor
lm.beta(lm2) # comparing standardized coefficients

predtestlm2<-predict(lm2,newdata=test) # predicting on test data
lm2values<-data.frame(obs=test$npo2008trans,pred=predtestlm2)
defaultSummary(lm2values) # summarizing RMSE, Rsquared, MAE
ggplot(lm2values,aes(x=pred,y=obs))+geom_point(size=2) + theme(text = element_text(size = 14))+geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +geom_abline(intercept = 0, slope = 1,color='red') # plot of obs vs pred and 1:1 line


# 10-fold cross validation (CV) on BIC-best linear regression
data_ctrl <- trainControl(method = "cv", number = 10)
model_caret <- train(npo2008trans ~ TBWDDN + RAXericRat,   # model to fit
                      data = data,                        
                      trControl = data_ctrl,              # folds
                      method = "lm"                      # specifying regression model
)

model_caret$finalModel
lm.beta(model_caret$finalModel)
model_caret$resample

# 10-fold CV with interaction term
data_ctrl <- trainControl(method = "cv", number = 10)
model_caret2 <- train(npo2008trans ~ TBWDDN * RAXericRat,   # model to fit
                      data = data,                        
                      trControl = data_ctrl,              # folds
                      method = "lm"                      # specifying regression model
)

model_caret2$finalModel
lm.beta(model_caret2$finalModel)
model_caret2$resample

# Final aspatial linear regression model with all data, visualizing the interaction
lm2allint<-lm(npo2008trans ~ TBWDDN * RAXericRat, data = data)
visreg(lm2allint)
visreg(lm2allint,"TBWDDN",by="RAXericRat",breaks=c(0,0.80))


# Transformations
atasmv1tr$npotr2<-log((datasmv1tr$npotr1*-1)+1) # example forward transformation from feet to transformed scale in R
backt<-function(actual) {result<-((exp(actual)-1)*-1);return(result)} # function for backtransforming random forest NPO predictions into standard units (i.e., feet)

# Regression kriging

datark<-data %>% dplyr::select(npo2008trans,TBWDDN,RAXericRat,XCOORD,YCOORD)
attach(datark) 
coordinates(datark) = ~ XCOORD + YCOORD # create SpatialPointsDataFrame object from data
vr1=autofitVariogram(npo2008trans  ~ 1+TBWDDN*RAXericRat, datark) # perform regression kriging, kriging residuals remaining after running regression with these three variables and their interactions
plot(vr1)
print(vr1)
reg_kriging_devdata=autoKrige(npo2008trans ~ 1+TBWDDN*RAXericRat, datark,datarkcopy) 
reg_kriging_devdata$krige_output@data$var1.pred
plot(reg_kriging_devdata)

kr.cv=autoKrige.cv(npo2008trans  ~ 1+TBWDDN*RAXericRat, datark) # perform a  leave one out cross-validation on linear regression kriging
cvrkvalues<-data.frame(obs=kr.cv$krige.cv_output@data$observed,pred=kr.cv$krige.cv_output@data$var1.pred)
defaultSummary(cvrkvalues) # cv performance summary
ggplot(cvrkvalues,aes(x=pred,y=obs))+geom_point(size=2) + theme(text = element_text(size = 14))+geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +geom_abline(intercept = 0, slope = 1,color='red') # plot of obs vs pred and 1:1 line

predictgridex <- read_csv("predictgridex.csv") # importing prediction grid to support kriging for whole study area instead of just points
coordinates(predictgridex) = ~ X_COORD + Y_COORD
reg_krig_studyarea=autoKrige(npo2008trans ~ 1+TBWDDN*RAXericRat, datark,predictgridex)
write.csv(reg_krig_studyarea$krige_output,file='reg_krig_studyarea.csv',row.names=F) # write out file for use in mapping in other software (e.g., ArcGIS)


# Random Forest

modata<-data %>% dplyr::select(npo2008trans, TBWDDN, RAXericRat, RAXericYN, ACRES_RA, AREAPERATI, Distnearwe, Headdiffer, Soilperm, IAthicknes, Rain10NN, Kerneld)
# Now Selecting 80% of data as sample from total 'n' rows of the data  
tempdata<-modata
set.seed(42)
sample <- sample.int(n = nrow(tempdata), size = floor(0.80*nrow(tempdata)), replace = F)
motrain <- tempdata[sample, ]
motest  <- tempdata[-sample, ]

rf1<-randomForest(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,data=motrain) # fitting random forest to training data

# optimizing hyperparameter mtry using caret
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(1:11))
metric<-"RMSE"
set.seed(9)
rf_gridsearch <- train(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,data=motrain, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

rf1m2n10<-randomForest(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,mtry=2,ntree=10000,data=motrain) # refit random forest with mtry=2 and ntree=10000

# evaluating optimized random forest on test data
predtestrf1m2n10<-predict(rf1m2n10,newdata=motest)
rf1values<-data.frame(obs=motest$npo2008trans,pred=predtestrf1m2n10)
defaultSummary(rf1values)
ggplot(rf1values,aes(x=pred,y=obs))+geom_point(size=2) + theme(text = element_text(size = 14))+geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +geom_abline(intercept = 0, slope = 1,color='red')

set.seed(9)
rf1m2n10all<-randomForest(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,mtry=2,ntree=10000,importance=TRUE,data=modata) # refit random forest on all data using optimized hyperparameters
print(rf1m2n10all)
varImpPlot(rf1m2n10all) # variable importance plot
partialPlot(rf1m2n10all,data.frame(modata), RAXericRat) # partial dependence plot
partialPlot(rf1m2n10all,data.frame(modata), TBWDDN) # partial dependence plot
partialPlot(rf1m2n10all,data.frame(modata), Rain10NN) # partial dependence plot

# evaluating performance of random forest with all data and optimized hyperparameters
modataeval<-data.frame(modata,pred=rf1m2n10all$predicted)
modataeval$obs=modataeval$npo2008trans
modataeval$resid<-modataeval$obs-modataeval$pred
ggplot(modataeval,aes(x=pred,y=obs))+geom_point(size=2) + theme(text = element_text(size = 14))+geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +geom_abline(intercept = 0, slope = 1,color='red')

# checking for residual autocorrelation in residuals from random forest
modataspat<-data.frame(modataeval,XCOORD=data$XCOORD,YCOORD=data$YCOORD)
coordinates(modataspat) = ~ XCOORD + YCOORD # create SpatialPointsDataFrame object from data
vrrf1=autofitVariogram(resid  ~ 1, modataspat) # perform regression kriging, kriging residuals remaining after running regression with these three variables and their interactions
plot(vrrf1)
print(vrrf1)

# ordinary kriging 
vrok=autofitVariogram(npo2008trans  ~ 1, modataspat) 
plot(vrok)
print(vrok)
kr.cv.ok=autoKrige.cv(npo2008trans  ~ 1, modataspat) # perform a leave one out cross validation for ordinary kriging
cvrkvaluesok<-data.frame(obs=kr.cv.ok$krige.cv_output@data$observed,pred=kr.cv.ok$krige.cv_output@data$var1.pred)
defaultSummary(cvrkvaluesok) # performance evaluation ordinary kriging
ggplot(cvrkvaluesok,aes(x=pred,y=obs))+geom_point(size=2) + theme(text = element_text(size = 14))+geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +geom_abline(intercept = 0, slope = 1,color='red')
ok_krig_studyarea=autoKrige(npo2008trans ~ 1, modataspat,predictgridex)
plot(ok_krig_studyarea)

# custom code to evaluate random forest kriging using loocv
set.seed(99)
xy<-modataspat # to be replaced with data to undergo LOOCV
n_train<-nrow(xy)
loocv_tmp <- matrix(NA, nrow = n_train)
for (k in 1:n_train) {
  train_xy <- xy[-k, ]
  test_xy <- xy[k, ]
  fitforest<-randomForest(npo2008trans~TBWDDN+RAXericRat+RAXericYN+ACRES_RA+AREAPERATI+Distnearwe+Headdiffer+Soilperm+IAthicknes+Rain10NN+Kerneld,mtry=2,ntree=10000,data=train_xy)
  predforest<-predict(fitforest,newdata=test_xy)
  residsforest<-train_xy$npo2008trans-predict(fitforest)
  train_xyappend<-train_xy
  train_xyappend$resids<-residsforest
  fitted_models<-autoKrige(residsforest ~ 1, train_xyappend,test_xy)
  predictions<-fitted_models$krige_output$var1.pred
  loocv_tmp[k]<-predictions
  loocv_tmp[k]<-loocv_tmp[k]+predforest
  cat(k)
  cat(" ")
}
cat("\n")
cat("r2 between loocv predications and actual:")
cor(loocv_tmp,xy$npo2008trans)^2
cat("complete")

cvpredsrfm2n10k<-loocv_tmp # saving the random forest loocv predictions

cvrfvalues<-data.frame(obs=xy$npo2008trans,pred=cvpredsrfm2n10k)
defaultSummary(cvrfvalues) # evaluating preformance of the random forest loocv predictions
ggplot(cvrfvalues,aes(x=pred,y=obs))+geom_point(size=2) + theme(text = element_text(size = 14))+geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +geom_abline(intercept = 0, slope = 1,color='red')

