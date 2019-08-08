## TODO: see caret::maxDissim()
splitdf<- function(df, seed=NULL, ratio=0.8) {
  if (!is.null(seed)) set.seed(seed)
  index<- 1:nrow(df)
  limitindex<- lapply(df, function(x){
    # limit<- range(x)
    mins<- which(x %in% min(x))
    maxs<- which(x %in% max(x))
    c(sample(mins, 1), sample(maxs, 1))
  })
  limitindex<- unique(unlist(limitindex))
  nTrain<- round(length(index) * ratio)

  if (length(limitindex) > nTrain){
    warning("The number of extrem cases is bigger than the number of cases used to train. Increase the train ratio of look for more data.")
    limitindex<- sample(limitindex, size=nTrain)
  }

  trainindex<- sample(setdiff(index, limitindex), size=nTrain - length(limitindex), replace=FALSE)
  trainindex<- c(limitindex, trainindex)
  trainset<- df[trainindex, ]
  testset<- df[-trainindex, ]
  list(trainset=trainset, testset=testset)
}
