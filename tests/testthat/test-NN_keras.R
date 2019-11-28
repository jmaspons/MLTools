context("NN_keras")

varScale<- seq(-100, 100, length.out=10)
names(varScale)<- paste0("X", 1:length(varScale))
df<- data.frame(lapply(varScale, function(i) runif(100) * i))
predInput<- data.frame(lapply(varScale, function(i) runif(50) * i))
responseVars<- 1
idVars<- character()
epochs<- 10
replicates<- 5
repVi<- 2
summarizePred<- TRUE
hidden_shape<- 10
batch_size<- "all"
tempdirRaster<- tempdir()
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
baseFilenameRasterPred<- paste0(tempdirRaster, "/testMap")
baseFilenameNN<- paste0(tempdir(), "/testNN")
DALEXexplainer<- TRUE
crossValRatio<- 0.7
NNmodel<- FALSE
verbose<- 0

test_that("process works", {
  system.time(res<- process(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                            baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  resB<- process(df=df, predInput=predInput, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                 DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
})

test_that("process_keras works", {
  # future::plan(future::multisession(workers=1))
  future::plan(future::transparent)
  # future::plan(future::sequential)
  system.time(res2<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                                   baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(res2B<- process_keras(df=df, predInput=predInput, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                        baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(res2reps<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
                                       hidden_shape=hidden_shape, batch_size=batch_size, summarizePred=FALSE, baseFilenameNN=baseFilenameNN,
                                       DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(res2Breps<- process_keras(df=df, predInput=predInput, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi,
                                        hidden_shape=hidden_shape, batch_size=batch_size, summarizePred=FALSE, baseFilenameNN=baseFilenameNN,
                                        DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
})

test_that("Predict with raster", {
  predInputR<- raster::raster(nrows=15, ncols=15)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  names(predInputR)<- names(df)
  # predInput<- predInputR


  resR<- process(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                 baseFilenameRasterPred=baseFilenameRasterPred, baseFilenameNN=baseFilenameNN,
                 DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  future::plan(future::sequential)
  future::plan(future::transparent)
  filenameRasterPred<- paste0(tempdir(), "/testMap1.grd") # avoid overwrite
  res2R<- process_keras(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                        filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  future::plan(future::multisession(workers=3))
  filenameRasterPred<- paste0(tempdir(), "/testMap2.grd") # avoid overwrite
  res3R<- process_keras(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                        summarizePred=FALSE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap3.grd") # avoid overwrite
  res3BR<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                        summarizePred=TRUE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap4.grd") # avoid overwrite
  res3repBR<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                         summarizePred=FALSE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                         DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
  # TODO:
  # res<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=10, replicates=5, repVi=2, batch_size="all",
  #               DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)
})


test_that("Future plans work", {
  future::plan(future::multisession(workers=3))
  # options(future.globals.onReference = "error")
  system.time(resSessions<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                          hidden_shape=hidden_shape, DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=FALSE, verbose=verbose))
# Error in keras::reset_states(modelNN) : attempt to apply non-function
# Don't import/export python objects to/from code inside future for PSOCK and multisession clusters
# https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html

  future::plan(future::transparent)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  future::plan(future::multicore)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  future::plan(future.callr::callr(workers=3))
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  future::plan(future::sequential)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
})

