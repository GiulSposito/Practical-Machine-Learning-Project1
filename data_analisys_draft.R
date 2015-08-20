## setup
library(caret)
library(ggplot2)
library(gridExtra)

## reading data
training <- read.csv("./data/pml-training.csv", na.strings=c("NA","","#DIV/0!"))
testing <- read.csv("./data/pml-testing.csv", na.strings=c("NA","","#DIV/0!"))

# exploring training data
str(training)

# subsetting the dataframeremoving columns with NA values for all row
nonNA_Cols <- sapply(training, function(x){mean(!is.na(x))}) # % of non NA values in a column
valid_Cols <- nonNA_Cols > 0 # T -> has some values or F -> only NA values 

# removing Non relevant data: shouldn't be predictors
valid_Cols[c("X", "user_name","cvtd_timestamp","num_window","new_window")]  <- F

# understanding the timestamp info
dt <- training[training$classe=="E" & training$user_name=="adelmo",] # just one user and class to see the behavior of timestamps

qplot(dt$raw_timestamp_part_1,dt$yaw_arm)
qplot(dt$raw_timestamp_part_2,dt$yaw_arm)
max(dt$raw_timestamp_part_2) # see the maximum value, so this is the dimention

# testing the sum of the two datasets
new_timestamp <- dt$raw_timestamp_part_1*1000000+dt$raw_timestamp_part_2
qplot(new_timestamp,dt$yaw_arm)

# adjusting the datasets
training_timestamp <- training$raw_timestamp_part_1*1000000+training$raw_timestamp_part_2
testing_timestamp <- testing$raw_timestamp_part_1*1000000+testing$raw_timestamp_part_2
valid_Cols[c("raw_timestamp_part_1","raw_timestamp_part_2")] <- F

# subsetting columns and adding the new timestamp info
training <- training[,which(valid_Cols==T)]
training$timestamp <- training_timestamp
testing <- testing[,which(valid_Cols==T)]
testing$timestamp <- testing_timestamp

# Near Zero Variance
nZeroVar <- nearZeroVar(training,saveMetrics=T)
nZeroVar[nZeroVar$nzv==T,]

# subsetting bya NZV
training <- training[,nZeroVar$nzv==F]
testing <- testing[,nZeroVar$nzv==F]

# partition in the training set (Cross Validation)
set.seed(1234)
inTrain <- createDataPartition(training$classe, p=.75, list=F)
train <- training[inTrain,]
cross <- training[-inTrain,]

# normalizing and Pre Processing
preProc <- preProcess(train[,-118],method=c("knnImpute", "center", "scale"))
pTrain <- predict(preProc,train[,-118])
pTrain$classe <- train$classe

# fitting
modFit <- train(classe~., method="rf", data=pTrain, allowParallel=T, ncores=4, nodesize=10, importance=F,  proximity=F, trControl=trainControl(method="cv"))
modFit

# cross validation
pCross <- predict(preProc,cross[,-118])
crossValid <- predict(modFit, pCross)
confusionMatrix(crossValid, cross$classe)

# predicting test set
pTest <- predict(preProc,testing[,-118])
result <- predict(modFit,pTest)
result
