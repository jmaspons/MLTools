context("xgboost")

varScale<- seq(-100, 100, length.out=4)
names(varScale)<- paste0("X", 1:length(varScale))
df<- data.frame(lapply(varScale, function(i) runif(100) * i), cat1=rep(LETTERS[1:5], times=20), cat2=rep(letters[1:10], each=10))
predInput<- data.frame(lapply(varScale, function(i) runif(50) * i), cat1=rep(LETTERS[1:5], times=10), cat2=rep(letters[1:10], each=5))
rowNames<- lapply(seq_len(nrow(df) %/% length(letters) + 1), function(x){
  sapply(letters, function(y) paste(rep(y, each=x), collapse=""))
})
rowNames<- do.call(c, rowNames)
rownames(df)<- rowNames[1:nrow(df)]
rownames(predInput)<- rowNames[1:nrow(predInput)]

responseVars<- 1
crossValStrategy<- c("Kfold", "bootstrap")
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
k<- 3
idVars<- character()
epochs<- 2
replicates<- 2
repVi<- 2
summarizePred<- TRUE
params<- list()
nrounds<- 2
scaleDataset<- FALSE
tempdirRaster<- tempdir()
dir.create(tempdirRaster, showWarnings=FALSE)
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
baseFilenameRasterPred<- paste0(tempdirRaster, "/testMap")
nCoresRaster<- 2
variableResponse<- TRUE
DALEXexplainer<- TRUE
save_validateset<- TRUE
XGBmodel<- TRUE
caseClass<- c(rep("A", 23), rep("B", 75), rep("C", 2))
weight<- "class"
verbose<- 2
verbose<- 0

test_that("pipe_xgboost works", {
  result<- list()
  set.seed(1)
  # future::plan(future::sequential, split=TRUE)
  future::plan(future::multisession)
  # future::plan(list(future::future::sequential, future::tweak(future::multisession, workers=2)))
  # future::plan(future.callr::callr(workers=3))
  # future::futureSessionInfo()
  system.time(result$resp1summarizedPred<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars,
                                                      epochs=epochs, repVi=repVi,
                                                      crossValStrategy=crossValStrategy[1], k=k, replicates=replicates,
                                                      params=params, nrounds=nrounds,
                                                      DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset,
                                                      crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))


  ## TODO: Fix warnings when rev(predInput) (?)
  system.time(result$resp1<- pipe_xgboost(df=df, predInput=rev(predInput), responseVars=responseVars,
                                        epochs=epochs, repVi=10,  # check names with 2 digits
                                        crossValStrategy=crossValStrategy[2], replicates=10,  # check names with 2 digits
                                        nrounds=nrounds, params=params, summarizePred=FALSE,
                                        DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset,
                                        crossValRatio=crossValRatio[1], XGBmodel=XGBmodel, verbose=verbose))


  tmp<- lapply(result, function(x) expect_s3_class(x, class="pipe_result.xgboost"))

  tmp<- lapply(result, function(x){
    expect_s3_class(x$performance, class="data.frame")
    reps<- nrow(x$performance)
    if (x$params$crossValStrategy == "bootstrap"){
      expect_equal(rownames(x$performance), expected=paste0("rep", formatC(1:reps, format="d", flag="0", width=nchar(reps))))
    } else if (x$params$crossValStrategy == "Kfold") {
      expect_equal(rownames(x$performance), expected=paste0("Fold", 2:k, ".Rep", rep(1:replicates, each=k-1))) # Fold2:k (Fold1 for validationset)
    }
  })

  tmp<- lapply(result, function(x){
    expect_type(x$scale, type="list")
    expect_equal(unique(lapply(x$scale, names)), expected=list(c("mean", "sd")))
  })

  tmp<- lapply(result, function(x){
    expect_s3_class(x$shap, class = "kernelshap")
  })

  tmp<- lapply(result, function(x){
    expect_type(x$vi, type="double")
    reps<- nrow(x$performance)
    repsVi<- x$params$repVi
    expectedColnames<- paste0(rep(paste0("rep", formatC(1:reps, format="d", flag="0", width=nchar(reps))), each=repsVi), "_",
                              rep(paste0("perm", formatC(1:repsVi, format="d", flag="0", width=nchar(repsVi))), times=reps))
    expect_equal(colnames(x$vi), expected=expectedColnames)
  })

  tmp<- lapply(result, function(x){
    lapply(x$variableResponse, expect_s3_class, class="partial_dependence_explainer")
    lapply(x$variableResponse, expect_s3_class, class="aggregated_profiles_explainer")
  })

  tmp<- lapply(result, function(x){
    expect_type(x$variableCoef, type="list")
    lapply(x$variableCoef, function(y){
      if (ncol(y) > 4){
        expectedColnames<- c("intercept", paste0("b", 1:(ncol(y) - 4)), "adj.r.squared", "r.squared", "degree")
      } else {
        expectedColnames<- c("intercept", "adj.r.squared", "r.squared", "degree")
      }
      expect_equal(colnames(y), expected=expectedColnames)
    })
  })

  tmp<- lapply(result, function(x){
    expect_type(x$predictions, type="double")
  })
  expectedColnames<- c(idVars, "Mean", "SD", "Naive SE", "2.5%", "25%", "50%", "75%", "97.5%")
  tmp<- expect_equal(colnames(result$resp1summarizedPred$predictions), expected=expectedColnames)
  tmp<- expect_equal(colnames(result$resp1$predictions), expected=c(idVars, paste0("rep", formatC(1:nrow(result$resp1$performance), format="d", flag="0", width=nchar(nrow(result$resp1$performance))))))

  tmp<- lapply(result, function(x){
    expect_type(x$model, type="list")
  })

  tmp<- lapply(result, function(x){
    expect_type(x$DALEXexplainer, type="list")
    reps<- nrow(x$performance)
    if (x$params$crossValStrategy == "bootstrap"){
      expect_equal(names(x$DALEXexplainer), expected=paste0("rep", formatC(1:reps, format="d", flag="0", width=nchar(reps))))
    } else if (x$params$crossValStrategy == "Kfold") {
      expect_equal(names(x$DALEXexplainer), expected=paste0("Fold", 2:k, ".Rep", rep(1:replicates, each=k-1)))  # Fold2:k (Fold1 for validationset)
    }
    lapply(x$DALEXexplainer, expect_s3_class, class="explainer")
  })

  tmp<- lapply(result, function(x){
    expect_type(x$validateset, type="list")
    reps<- nrow(x$performance)
    if (x$params$crossValStrategy == "bootstrap"){
      expect_equal(names(x$validateset), expected=paste0("rep", formatC(1:reps, format="d", flag="0", width=nchar(reps))))
    } else if (x$params$crossValStrategy == "Kfold") {
      expect_equal(names(x$validateset), expected=paste0("Fold", 2:k, ".Rep", rep(1:replicates, each=k-1)))  # Fold2:k (Fold1 for validationset)
    }
    lapply(x$validateset, expect_s3_class, class="data.frame")
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

  ## TODO: categorical variables for rasters
  df<- df[, names(predInputR)]

  suppressWarnings(future::plan(future::multicore))
  # DEBUG: future::plan(future::sequential, split=TRUE)
  filenameRasterPred<- paste0(tempdir(), "/testMap1.grd") # avoid overwrite
  resultR$summarizedPred<- pipe_xgboost(df, predInput=predInputR,
                                           epochs=epochs, repVi=repVi,
                                           crossValStrategy=crossValStrategy[1], k=k, replicates=replicates,
                                           params=params, nrounds=nrounds, summarizePred=TRUE,
                                           filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster,
                                           DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap2.grd") # avoid overwrite
  resultR$pred<- pipe_xgboost(df, predInput=predInputR[[rev(names(predInputR))]],
                             epochs=epochs, repVi=repVi,
                             crossValStrategy=crossValStrategy[2], replicates=replicates,
                             params=params, nrounds=nrounds, summarizePred=FALSE,
                             filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster,
                             DALEXexplainer=FALSE, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose)

  resultR$inMemory<- pipe_xgboost(df, predInput=predInputR,
                                                  epochs=epochs, repVi=repVi,
                                                  crossValStrategy=crossValStrategy[1], k=k, replicates=replicates,
                                                  params=params, nrounds=nrounds, summarizePred=TRUE,
                                                  tempdirRaster=tempdirRaster,
                                                  DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose)


  tmp<- lapply(resultR, function(x){
    expect_s4_class(x$predictions, class="Raster")
  })
  tmp<- expect_equal(names(resultR$summarizedPred$predictions), expected=c("mean", "sd", "se"))
  tmp<- expect_equal(names(resultR$pred$predictions), expected=paste0("X1_rep", 1:replicates))
  tmp<- expect_equal(names(resultR$inMemory$predictions), expected=c("mean", "sd", "se"))

  # lapply(resultR, function(x) names(x$predictions))
  ## Check NAs position
  # raster::plot(predInputR)
  # raster::plot(resultR$summarizedPred$predictions)
  # raster::plot(resultR$pred$predictions)
  # raster::plot(resultR$inMemory$predictions)

  file.remove(dir(tempdir(), "testMap.+\\.gr(i|d)$", full.names=TRUE))
})


test_that("Future plans work", {
  # options(future.globals.onReference = "error")
  # Don't import/export python objects to/from code inside future for PSOCK and callR clusters
  # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html

  future::plan(future::sequential, split=TRUE)
  system.time(res<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, params=params,
                               nrounds=nrounds, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.xgboost")

  future::plan(future::multicore)
  system.time(res<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, params=params,
                               nrounds=nrounds, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.xgboost")

  future::plan(future.callr::callr(workers=3))
  system.time(res<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, params=params,
                               nrounds=nrounds, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.xgboost")

  future::plan(future::sequential)
  system.time(res<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, params=params,
                               nrounds=nrounds, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.xgboost")
})


test_that("scaleDataset", {
  future::plan(future::multisession)
  system.time(res<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi,
                               params=params, scaleDataset=TRUE, nrounds=nrounds,
                               DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.xgboost")

  predInputR<- raster::raster(nrows=15, ncols=15)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  names(predInputR)<- names(varScale)
  # predInput<- predInputR

  ## TODO: categorical variables for rasters
  df<- df[, names(predInputR)]

  filenameRasterPred<- paste0(tempdir(), "/testMapScaleDataset.grd") # avoid overwrite
  res<- pipe_xgboost(df, predInput=predInputR, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, params=params,
                   scaleDataset=TRUE,  nrounds=nrounds,
                   filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster,
                   DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose)
  expect_s3_class(res, class="pipe_result.xgboost")
})


test_that("summary", {
  future::plan(future::multisession)
  system.time(res<- pipe_xgboost(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi,
                               params=params, scaleDataset=TRUE, nrounds=nrounds,
                               DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, XGBmodel=XGBmodel, verbose=verbose))

  sres<- summary(res)
  expect_s3_class(sres, class="summary.pipe_result.xgboost")
  expect_type(sres, type="list")
})

