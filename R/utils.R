## TODO: see caret::maxDissim()
splitdf<- function(df, ratio=0.7, trainLimits=TRUE, seed) {
  if (!missing(seed)) set.seed(seed)
  index<- 1:nrow(df)

  if (trainLimits){
    limitindex<- apply(df[, apply(df, 2, is.numeric)], 2, function(x){
      # limit<- range(x)
      mins<- which(x %in% min(x))
      maxs<- which(x %in% max(x))
      c(sample(mins, 1), sample(maxs, 1))
    })
    limitindex<- unique(as.vector(unlist(limitindex)))
  } else {
    limitindex<- integer()
  }

  nTrain<- round(length(index) * ratio)

  if (length(limitindex) > nTrain){
    warning("The number of extrem cases is bigger than the number of cases used to train. Increase the train ratio of look for more data.")
    limitindex<- sample(limitindex, size=nTrain)
  }

  trainindex<- sample(setdiff(index, limitindex), size=nTrain - length(limitindex), replace=FALSE)
  trainindex<- c(limitindex, trainindex)
  trainset<- df[trainindex, ]
  testset<- df[-trainindex, ]

  return(list(trainset=trainset, testset=testset))
}

summarize_pred<- function(pred, ...) UseMethod("summarize_pred", pred)

summarize_pred.default<- function(pred){
  t(apply(pred, 1, function(x){
        pred.summary<- summary(coda::mcmc(x))
        c(pred.summary$statistics[c("Mean", "SD", "Naive SE")], pred.summary$quantiles)
      }))
}

summarize_pred.Raster<- function(pred, filename, nCoresRaster=parallel::detectCores() %/% 2){
  raster::beginCluster(n=nCoresRaster)

  meanMap<- raster::clusterR(pred, function(x){
    raster::calc(x, fun=mean)
  })

  sdMap<- raster::clusterR(pred, function(x){
    raster::calc(x, fun=stats::sd)
  })

  sqrtN<- sqrt(raster::nlayers(pred))

  seMap<- raster::clusterR(sdMap, function(x){
    raster::calc(x, fun=function(y) y / sqrtN)
  })

  raster::endCluster()

  out<- raster::stack(list(mean=meanMap, sd=sdMap, se=seMap))

  if (!missing(filename)){
    out<- raster::brick(out, filename=filename)

    ## Remove temporal files
    tmpFiles<- sapply(list(mean=meanMap, sd=sdMap, se=seMap), function(x){
      raster::filename(x)
    })

    tmpFiles<- tmpFiles[tmpFiles != ""]
    file.remove(tmpFiles, gsub("\\.grd", ".gri", tmpFiles))
  }

  out
}
