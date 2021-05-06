context("NN_keras")

varScale<- seq(-100, 100, length.out=4)
names(varScale)<- paste0("X", 1:length(varScale))
df<- data.frame(lapply(varScale, function(i) runif(100) * i))
predInput<- data.frame(lapply(varScale, function(i) runif(50) * i))
responseVars<- 1
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
idVars<- character()
epochs<- 5
replicates<- 2
repVi<- 3
summarizePred<- TRUE
hidden_shape<- 3
batch_size<- "all"
scaleDataset<- FALSE
tempdirRaster<- tempdir()
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
baseFilenameRasterPred<- paste0(tempdirRaster, "/testMap")
baseFilenameNN<- paste0(tempdir(), "/testNN")
variableResponse<- TRUE
DALEXexplainer<- TRUE
NNmodel<- TRUE
verbose<- 0
caseClass<- c(rep("A", 23), rep("B", 77))
weight<- "class"


test_that("process_keras works", {
  result<- list()

  # future::plan(future::transparent)
  future::plan(future::multisession)
  system.time(result$resp1summarizedPred<- process_keras(df=df, predInput=predInput, responseVars=responseVars,
                                                         epochs=epochs, replicates=replicates, repVi=repVi,
                                                         batch_size=batch_size, hidden_shape=hidden_shape,
                                                         baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer,
                                                         crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(result$resp2summarizedPred<- process_keras(df=df, predInput=predInput, responseVars=1:2,
                                                         epochs=epochs, replicates=replicates, repVi=repVi,
                                                         batch_size=batch_size, hidden_shape=hidden_shape,
                                                         baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer,
                                                         crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(result$resp1<- process_keras(df=df, predInput=rev(predInput), responseVars=responseVars,
                                           epochs=epochs, replicates=replicates, repVi=repVi,
                                           hidden_shape=hidden_shape, batch_size=batch_size, summarizePred=FALSE,
                                           baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer,
                                           crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(result$resp2<- process_keras(df=df, predInput=rev(predInput), responseVars=1:2,
                                           epochs=epochs, replicates=replicates, repVi=repVi,
                                           hidden_shape=hidden_shape, batch_size=batch_size, summarizePred=FALSE,
                                           baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer,
                                           crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  tmp<- lapply(result, function(x) expect_s3_class(x, class="process_NN"))

  tmp<- lapply(result, function(x){
      expect_s3_class(x$performance, class="data.frame")
      expect_equal(rownames(x$performance), expected=paste0("rep", 1:replicates))
    })

  tmp<- lapply(result, function(x){
    expect_type(x$scale, type="list")
    expect_equal(unique(lapply(x$scale, names)), expected=list(c("mean", "sd")))
  })

  expectedColnames<- paste0(rep(paste0("rep", formatC(1:replicates, format="d", flag="0", width=nchar(replicates))), each=repVi + 1), "_",
                              rep(paste0("perm", formatC(0:repVi, format="d", flag="0", width=nchar(repVi))), times=replicates))
  tmp<- lapply(result, function(x){
    expect_type(x$vi, type="double")
    expect_equal(colnames(x$vi), expected=expectedColnames)
  })

  tmp<- lapply(result, function(x){
    lapply(x$variableResponse, expect_s3_class, class="partial_dependence_explainer")
    lapply(x$variableResponse, expect_s3_class, class="aggregated_profiles_explainer")
  })

  tmp<- lapply(result, function(x){
    expect_type(x$variableCoef, type="list")
    lapply(x$variableCoef, function(y){
      expectedColnames<- c("intercept", paste0("b", 1:(ncol(y) - 4)), "adj.r.squared", "r.squared", "degree")
      expect_equal(colnames(y), expected=expectedColnames)
    })
  })

  tmp<- lapply(result, function(x){
    expect_type(x$predictions, type="list")
    expect_type(x$predictions[[1]], type="double")
  })
  expectedColnames<- c("Mean", "SD", "Naive SE", "2.5%", "25%", "50%", "75%", "97.5%")
  tmp<- expect_equal(colnames(result$resp1summarizedPred$predictions[[1]]), expected=expectedColnames)
  tmp<- expect_equal(unlist(unique(lapply(result$resp2summarizedPred$predictions, colnames))), expected=expectedColnames)
  tmp<- expect_equal(colnames(result$resp1$predictions[[1]]), expected=paste0("rep", 1:replicates))
  tmp<- expect_equal(unlist(unique(lapply(result$resp2$predictions, colnames))), expected=paste0("rep", 1:replicates))

  tmp<- lapply(result, function(x){
    expect_type(x$model, type="list")
    lapply(x$model, function(y) expect_type(y, type="raw"))
    lapply(x$model, function(y) expect_s3_class(keras::unserialize_model(y), class="keras.engine.sequential.Sequential"))
  })
  # dir(tempdir(), full.names=TRUE)
  expect_true(any(grepl(baseFilenameNN, dir(tempdir(), full.names=TRUE))))
  expect_equal(sum(grepl(baseFilenameNN, dir(tempdir(), full.names=TRUE))), replicates)

  tmp<- lapply(result, function(x){
    expect_type(x$DALEXexplainer, type="list")
    expect_equal(names(x$DALEXexplainer), expected=paste0("rep", 1:replicates))
    lapply(x$DALEXexplainer, expect_s3_class, class="explainer")
  })
})


test_that("Predict with raster", {
  predInputR<- raster::raster(nrows=4, ncols=6)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  # Put some NAs to detect rotations
  NAs<- expand.grid(col=1:ncol(predInputR), row=1:nrow(predInputR))
  NAs<- NAs[NAs$row > NAs$col, ]
  predInputR[NAs$row, NAs$col]<- NA

  names(predInputR)<- names(varScale)
  resultR<- list()
  # predInput<- predInputR


  future::plan(future::multiprocess)
  filenameRasterPred<- paste0(tempdir(), "/testMap1.grd") # avoid overwrite
  resultR$resp1summarizedPred<- process_keras(df, predInput=predInputR,
                                              epochs=epochs, replicates=replicates, repVi=repVi,
                                              batch_size=batch_size, hidden_shape=hidden_shape,
                                              filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                                              DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap2.grd") # avoid overwrite
  resultR$resp1<- process_keras(df, predInput=predInputR[[rev(names(predInputR))]],
                                epochs=epochs, replicates=replicates, repVi=repVi,
                                batch_size=batch_size, hidden_shape=hidden_shape, summarizePred=FALSE,
                                filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                                DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap3.grd") # avoid overwrite
  resultR$resp2summarizedPred<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                        summarizePred=TRUE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap4.grd") # avoid overwrite
  resultR$resp2<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size, hidden_shape=hidden_shape,
                         summarizePred=FALSE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                         DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  tmp<- lapply(resultR, function(x){
    expect_s4_class(x$predictions, class="RasterBrick")
  })
  tmp<- expect_equal(names(resultR$resp1summarizedPred$predictions), expected=c("mean", "sd", "se"))
  tmp<- expect_equal(names(resultR$resp1$predictions), expected=paste0("X1_rep", 1:replicates))

  # lapply(resultR, function(x) names(x$predictions))
  ## Check NAs position
  # plot(predInputR)
  # plot(resultR$resp1summarizedPred$predictions)
  # plot(resultR$resp1$predictions)
  # plot(resultR$resp2summarizedPred$predictions)
  # plot(resultR$resp2$predictions)

  file.remove(dir(tempdir(), "testMap.+\\.gr(i|d)$", full.names=TRUE))
})


test_that("Future plans work", {
  # options(future.globals.onReference = "error")
# Error in keras::reset_states(modelNN) : attempt to apply non-function
# Don't import/export python objects to/from code inside future for PSOCK and callR clusters
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


test_that("scaleDataset", {
  future::plan(future::multiprocess)
  system.time(res1<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
                                   batch_size=batch_size, scaleDataset=TRUE, hidden_shape=hidden_shape,
                                   baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  predInputR<- raster::raster(nrows=15, ncols=15)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  names(predInputR)<- names(df)
  # predInput<- predInputR

  filenameRasterPred<- paste0(tempdir(), "/testMapScaleDataset.grd") # avoid overwrite
  res2R<- process_keras(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                        scaleDataset=TRUE,  hidden_shape=hidden_shape,
                        filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
})


test_that("summary", {
  future::plan(future::multisession)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
                                   batch_size=batch_size, scaleDataset=TRUE, hidden_shape=hidden_shape,
                                   baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  sres<- summary(res)
  expect_s3_class(sres, class="summary.process_NN")
  expect_type(sres, type="list")
})

