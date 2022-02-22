context("LSTM_keras")

tensorflow::tf$get_logger()$setLevel("ERROR")

df<- data.frame(id=rep(LETTERS[1:10], each=5), static=rep(1:10, each=5), time=rep(1:5, times=5))
df$x1<- df$time * df$static
df$x2<- rnorm(nrow(df), mean=df$time * df$static + 100, sd=5)
df$x3<- rnorm(nrow(df), mean=df$time * df$static * 3, sd=2)
df$y<- rnorm(nrow(df), mean=(df$x1 + df$x2) / df$x3, sd=2)

timevar<- "time"
idVars<- "id"
responseVars<- "y"
staticVars<- "static"
predTemp<- c("x1", "x2", "x3")
predInput<- df
# df<- df[, c(idCols, timevar, responseVars, predInput)]
# id.vars<- c("id_1", "id_2", "static")
# vars<- setdiff(names(dl), c(id.vars, "time"))
# vars.ts<- setdiff(names(dw), c(id.vars, "time"))

predInput=predInput; responseVars=responseVars; caseClass=NULL; idVars=idVars; weight="class";
modelType="LSTM"; timevar=timevar; responseTime="LAST"; regex_time="[0-9]+"; staticVars=staticVars;
repVi=5; crossValStrategy="Kfold"; k=5; replicates=10; crossValRatio=c(train=0.6, test=0.2, validate=0.2);
hidden_shape=50; epochs=500; maskNA=-999; batch_size=1000;
summarizePred=TRUE; scaleDataset=FALSE; NNmodel=FALSE; DALEXexplainer=FALSE; variableResponse=FALSE;
baseFilenameNN=NULL; filenameRasterPred=NULL; tempdirRaster=NULL; nCoresRaster=parallel::detectCores() %/% 2; verbose=2

## TODO: FIX IT ----
repVi=0; DALEXexplainer=FALSE; variableResponse=FALSE
future::plan(future::transparent)
res<- pipe_keras_timeseries(df=df, predInput=predInput, responseVars=responseVars, caseClass=NULL, idVars=idVars, weight="class",
                            modelType="LSTM", timevar=timevar, responseTime="LAST", regex_time="[0-9]+", staticVars=staticVars,
                            repVi=repVi, crossValStrategy="Kfold", k=5, replicates=10, crossValRatio=c(train=0.6, test=0.2, validate=0.2),
                            hidden_shape=50, epochs=500, maskNA=-999, batch_size=50000,
                            summarizePred=TRUE, scaleDataset=FALSE, NNmodel=FALSE, DALEXexplainer=FALSE, variableResponse=FALSE,
                            baseFilenameNN=NULL, filenameRasterPred=NULL, tempdirRaster=NULL, nCoresRaster=parallel::detectCores() %/% 2, verbose=2)

crossValStrategy<- c("Kfold", "bootstrap")
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
k<- 3
epochs<- 5
maskNA<- -999
replicates<- 2
repVi<- 3
summarizePred<- TRUE
hidden_shape<- 3
batch_size<- "all"
scaleDataset<- FALSE
tempdirRaster<- tempdir()
dir.create(tempdirRaster, showWarnings=FALSE)
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
baseFilenameRasterPred<- paste0(tempdirRaster, "/testMap")
baseFilenameNN<- paste0(tempdir(), "/testNN")
nCoresRaster<- 2
variableResponse<- TRUE
DALEXexplainer<- TRUE
NNmodel<- TRUE
verbose<- 0
caseClass<- c(rep("A", 23), rep("B", 75), rep("C", 2))
weight<- "class"


test_that("process_keras works", {
  tensorflow::tf$get_logger()$setLevel("ERROR")
  result<- list()

  # future::plan(future::transparent)
  future::plan(future::multisession)
  # future::futureSessionInfo()
  # f<- future::future(tensorflow::tf$get_logger()$setLevel("ERROR"))
  # future::value(f)
  system.time(result$resp1summarizedPred<- process_keras(df=df, predInput=predInput, responseVars=responseVars,
                                                         epochs=epochs, repVi=repVi,
                                                         crossValStrategy=crossValStrategy[1], k=k,
                                                         batch_size=batch_size, hidden_shape=hidden_shape,
                                                         baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse,
                                                         crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(result$resp2summarizedPred<- process_keras(df=df, predInput=predInput, responseVars=1:2,
                                                         epochs=epochs, maskNA=maskNA, repVi=repVi,
                                                         crossValStrategy=crossValStrategy[2], replicates=replicates,
                                                         batch_size=batch_size, hidden_shape=hidden_shape,
                                                         baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse,
                                                         crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  system.time(result$resp1<- process_keras(df=df, predInput=rev(predInput), responseVars=responseVars,
                                           epochs=epochs, maskNA=maskNA, repVi=repVi,
                                           crossValStrategy=crossValStrategy[2], replicates=replicates,
                                           hidden_shape=hidden_shape, batch_size=batch_size, summarizePred=FALSE,
                                           baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse,
                                           crossValRatio=crossValRatio[1], NNmodel=NNmodel, verbose=verbose))

  system.time(result$resp2<- process_keras(df=df, predInput=rev(predInput), responseVars=1:2,
                                           epochs=epochs, repVi=repVi,
                                           crossValStrategy=crossValStrategy[1], k=k,
                                           hidden_shape=hidden_shape, batch_size=batch_size, summarizePred=FALSE,
                                           baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse,
                                           crossValRatio=c(train=0.8, test=0.2), NNmodel=NNmodel, verbose=verbose))

  tmp<- lapply(result, function(x) expect_s3_class(x, class="process_NN"))

  tmp<- lapply(result, function(x){
      expect_s3_class(x$performance, class="data.frame")
      reps<- nrow(x$performance)
      expect_equal(rownames(x$performance), expected=paste0("rep", 1:reps))
    })

  tmp<- lapply(result, function(x){
    expect_type(x$scale, type="list")
    expect_equal(unique(lapply(x$scale, names)), expected=list(c("mean", "sd")))
  })

  tmp<- lapply(result, function(x){
    expect_type(x$vi, type="double")
    reps<- nrow(x$performance)
    expectedColnames<- paste0(rep(paste0("rep", formatC(1:reps, format="d", flag="0", width=nchar(reps))), each=repVi + 1), "_",
                              rep(paste0("perm", formatC(0:repVi, format="d", flag="0", width=nchar(repVi))), times=reps))
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
  tmp<- expect_equal(colnames(result$resp1$predictions[[1]]), expected=paste0("rep", 1:nrow(result$resp1$performance)))
  tmp<- expect_equal(unlist(unique(lapply(result$resp2$predictions, colnames))), expected=paste0("rep", 1:nrow(result$resp2$performance)))

  tmp<- lapply(result, function(x){
    expect_type(x$model, type="list")
    lapply(x$model, function(y) expect_type(y, type="raw"))
    lapply(x$model, function(y) expect_s3_class(keras::unserialize_model(y), class="keras.engine.sequential.Sequential"))
  })
  # dir(tempdir(), full.names=TRUE)
  expect_true(any(grepl(baseFilenameNN, dir(tempdir(), full.names=TRUE))))
  expect_equal(sum(grepl(baseFilenameNN, dir(tempdir(), full.names=TRUE))), k - 1)

  tmp<- lapply(result, function(x){
    expect_type(x$DALEXexplainer, type="list")
    reps<- nrow(x$performance)
    expect_equal(names(x$DALEXexplainer), expected=paste0("rep", 1:reps))
    lapply(x$DALEXexplainer, expect_s3_class, class="explainer")
  })
})


test_that("Predict with raster", {
  tensorflow::tf$get_logger()$setLevel("ERROR")
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


  suppressWarnings(future::plan(future::multicore))
  filenameRasterPred<- paste0(tempdir(), "/testMap1.grd") # avoid overwrite
  resultR$resp1summarizedPred<- process_keras(df, predInput=predInputR,
                                              epochs=epochs, repVi=repVi,
                                              crossValStrategy=crossValStrategy[1], k=k,
                                              batch_size=batch_size, hidden_shape=hidden_shape, summarizePred=TRUE,
                                              filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                                              DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap2.grd") # avoid overwrite
  resultR$resp1<- process_keras(df, predInput=predInputR[[rev(names(predInputR))]],
                                epochs=epochs, maskNA=maskNA, repVi=repVi,
                                crossValStrategy=crossValStrategy[2], replicates=replicates,
                                batch_size=batch_size, hidden_shape=hidden_shape, summarizePred=FALSE,
                                filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                                DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap3.grd") # avoid overwrite
  resultR$resp2summarizedPred<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, maskNA=maskNA, repVi=repVi,
                        crossValStrategy=crossValStrategy[1], k=k, batch_size=batch_size, hidden_shape=hidden_shape,
                        summarizePred=TRUE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap4.grd") # avoid overwrite
  resultR$resp2<- process_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, repVi=repVi, replicates=replicates, batch_size=batch_size, hidden_shape=hidden_shape,
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
  tensorflow::tf$get_logger()$setLevel("ERROR")
  # options(future.globals.onReference = "error")
# Error in keras::reset_states(modelNN) : attempt to apply non-function
# Don't import/export python objects to/from code inside future for PSOCK and callR clusters
# https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html

  suppressWarnings(future::plan(future::transparent))
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
  expect_s3_class(res, class="process_NN")

  future::plan(future::multicore)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
  expect_s3_class(res, class="process_NN")

  future::plan(future.callr::callr(workers=3))
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
  expect_s3_class(res, class="process_NN")

  future::plan(future::sequential)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                  hidden_shape=hidden_shape, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
  expect_s3_class(res, class="process_NN")
})


test_that("scaleDataset", {
  tensorflow::tf$get_logger()$setLevel("ERROR")
  future::plan(future::multisession)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
                                   batch_size=batch_size, scaleDataset=TRUE, hidden_shape=hidden_shape,
                                   baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
  expect_s3_class(res, class="process_NN")

  predInputR<- raster::raster(nrows=15, ncols=15)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  names(predInputR)<- names(df)
  # predInput<- predInputR

  filenameRasterPred<- paste0(tempdir(), "/testMapScaleDataset.grd") # avoid overwrite
  res<- process_keras(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                        scaleDataset=TRUE,  hidden_shape=hidden_shape,
                        filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
                        DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
  expect_s3_class(res, class="process_NN")
})


test_that("summary", {
  tensorflow::tf$get_logger()$setLevel("ERROR")
  future::plan(future::multisession)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
                                   batch_size=batch_size, scaleDataset=TRUE, hidden_shape=hidden_shape,
                                   baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  sres<- summary(res)
  expect_s3_class(sres, class="summary.process_NN")
  expect_type(sres, type="list")
})

