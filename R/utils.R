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


# https://github.com/rspatial/raster/blob/b1c9d91b1b43b17ea757889dc93f97bd70dc1d2e/R/predict.R
?raster::`predict,Raster-method`
?predict.keras.engine.training.Model

predict.Raster_keras<- function(object, model, filename, fun=predict, ...) {
  out<- raster::raster(object)
  big<- !raster::canProcessInMemory(out, raster::nlayers(object) + 1)
  filename<- raster::trim(filename)

  if (big & filename == "") {
    filename<- raster::rasterTmpFile()
  }

  if (filename != "") {
    out<- raster::writeStart(out, filename)
    todisk<- TRUE
  } else {
    vv<- matrix(ncol=nrow(out), nrow=ncol(out))
    todisk<- FALSE
  }

  bs<- raster::blockSize(object)
  pb<- raster::pbCreate(bs$n)

  if (todisk) {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- predict(object=model, v, ...)

      out<- raster::writeValues(out, v, bs$row[i])
      raster::pbStep(pb, i)
    }

    out<- raster::writeStop(out)
  } else {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- predict(object=model, v, ...)

      cols<- bs$row[i]:(bs$row[i] + bs$nrows[i] - 1)
      vv[,cols]<- matrix(v, nrow=out@ncols)

      raster::pbStep(pb, i)
    }

    out<- raster::setValues(out, as.vector(vv))
  }

  raster::pbClose(pb)

  return(out)
}
