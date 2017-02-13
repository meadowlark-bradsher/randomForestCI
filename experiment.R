library(devtools)

install_github("meadowlark-bradsher/randomForestCI")

library(randomForest)
library(randomForestCI)
library(pROC)


#
# Spambase example
#

spam.data = read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
                     header = FALSE)
source("https://raw.githubusercontent.com/ggrothendieck/gsubfn/master/R/list.R")
devtools::install_github("ggrothendieck/gsubfn")
library(gsubfn)

# Separate into predictors and response
X = spam.data[, 1:57]
Y = spam.data[, 58]

rndSamp = sample(seq(1,4601),3000)
trainSamp = spam.data[rndSamp,]
testSamp = spam.data[-rndSamp,]

X = trainSamp[, 1:57]
Y = trainSamp[, 58]

newData = testSamp[, 1:57]
truth = testSamp[, 58]

# With 2000 trees, training takes longer, but the variance estimates are much better
rf.spam = randomForest(X, factor(Y), keep.inbag = TRUE, ntree = 2000)
ij.spam = randomForestInfJack(rf.spam, X, calibrate = TRUE)
plot(ij.spam)

spam.cpred = predict(rf.spam, newData, type="class")
cm = as.matrix(table(observed=truth, predicted=spam.cpred))
correct = sum(diag(cm))
n = sum(cm)


quantileRange = seq(0.05,0.95, by=(0.05))
quantiles = quantile(ij.spam[,2],probs=quantileRange)

getAccuracyOnThreshold <- function(x){

  conRows = which(ij.spam[,2] < quantiles[x])
  X1 = trainSamp[conRows, 1:57]
  Y1 = trainSamp[conRows, 58]

  # With 2000 trees, training takes longer, but the variance estimates are much better
  rf.spam = randomForest(X1, factor(Y1), keep.inbag = TRUE, ntree = 2000)
  spam.cpred = predict(rf.spam, newData, type="class")
  spam.ppred = predict(rf.spam, newData, type="prob")

  cm = as.matrix(table(observed=truth, predicted=spam.cpred))
  correct = sum(diag(cm))
  n = sum(cm)

  correct/n
  #as.numeric(roc(truth, spam.ppred[,1])$auc)
}

accuracies = sapply(seq(1:length(quantiles)),getAccuracyOnThreshold)

plot(accuracies~quantileRange, type='l')
#plot(AUCs~quantileRange, type='l')


cols = dim(X)[2]
rows = dim(X)[1]
randomIndependent = matrix(0,nrow=rows, ncol=cols)
randomDependent = sample(c(0,1),rows, replace=TRUE)

for(i in 1:cols){
  maxVal = max(X[,i])
  minVal = min(X[,i])
  for(j in 1:rows){
    randomIndependent[j,i] = runif(1,min=minVal, max=maxVal)
  }
}

toShuffleDeck1 = as.matrix(cbind(randomIndependent, randomDependent))
toShuffleDeck2 = as.matrix(trainSamp)
names(toShuffleDeck1) <- names(toShuffleDeck2)
toShuffleDeck = rbind(toShuffleDeck2, toShuffleDeck1)

shuffleDeck = toShuffleDeck[sample(nrow(toShuffleDeck)),]

shuffleX = shuffleDeck[, 1:57]
shuffleY = shuffleDeck[, 58]

# With 2000 trees, training takes longer, but the variance estimates are much better
rf.spam2 = randomForest(shuffleX, factor(shuffleY), keep.inbag = TRUE, ntree = 2000, mtry=19)
ij.spam2 = randomForestInfJack(rf.spam2, shuffleX, calibrate = TRUE)
plot(ij.spam2)

# With 2000 trees, training takes longer, but the variance estimates are much better
rf.spam3 = randomForest(randomIndependent, factor(randomDependent), keep.inbag = TRUE, ntree = 2000)
ij.spam3 = randomForestInfJack(rf.spam3, randomIndependent, calibrate = TRUE)
plot(ij.spam3)

spam.cpred = predict(rf.spam2, newData, type="class")
#cm = as.matrix(table(observed=truth, predicted=spam.cpred))

quantiles2 = quantile(ij.spam2[,2],probs=quantileRange)

AUCs2 <- rep(0,19)
accuracies2 <- rep(0,19)

getAccuracyOnThreshold <- function(x){

  conRows = which(ij.spam2[,2] < quantiles2[x])
  X1 = shuffleDeck[conRows, 1:57]
  Y1 = shuffleDeck[conRows, 58]

  # With 2000 trees, training takes longer, but the variance estimates are much better
  rf.spam.test = randomForest(X1, factor(Y1), keep.inbag = TRUE, ntree = 2000, mtry=19)

  spam.cpred = predict(rf.spam.test, newData, type="class")
  cm = as.matrix(table(observed=truth, predicted=spam.cpred))
  correct = sum(diag(cm))
  n = sum(cm)
  correct/n
}

accuracies2 = sapply(seq(1:length(quantiles2)),getAccuracyOnThreshold)
plot(accuracies2~quantileRange, type='l')
#plot(AUCs2~quantileRange, type='l')


