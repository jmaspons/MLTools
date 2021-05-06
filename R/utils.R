
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
