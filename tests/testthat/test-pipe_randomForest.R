context("randomForest")

df <- na.omit(airquality)
responseVar <- "Ozone"
predInput <- df
varScale<- seq(-100, 100, length.out=ncol(df))
names(varScale)<- names(df)

dfCat <- iris #[iris$Species %in% c("setosa", "versicolor"), ] # Only 2 categories supported
dfCat$Species <- as.character(dfCat$Species)
responseVarCat <- "Species"
predInputCat <- dfCat

crossValStrategy<- c("Kfold", "bootstrap")
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
k<- 3
idVars<- character()
replicates<- 2
repVi<- 2
summarizePred<- TRUE
shap<- TRUE
ntree<- 10
importance<- TRUE
scaleDataset<- FALSE
tempdirRaster<- tempdir()
dir.create(tempdirRaster, showWarnings=FALSE)
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
baseFilenameRasterPred<- paste0(tempdirRaster, "/testMap")
nCoresRaster<- 2
variableResponse<- TRUE
DALEXexplainer<- TRUE
save_validateset<- TRUE
RFmodel<- TRUE
caseClass<- c(rep("A", 23), rep("B", 75), rep("C", 13)) ## TODO: use it on tests!
caseClassCat<- dfCat$Species ## TODO: use it on tests!
weight<- "class"
verbose<- 2
verbose<- 0

test_that("pipe_randomForest works", {
  result<- list()
  set.seed(1)
  # future::plan(future::sequential, split=TRUE)
  future::plan(future::multisession)
  # future::plan(list(future::future::sequential, future::tweak(future::multisession, workers=2)))
  # future::plan(future.callr::callr(workers=3))
  # future::futureSessionInfo()
  system.time(result$resp1summarizedPred<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar,
                                                      repVi=10, # check names with 2 digits
                                                      crossValStrategy=crossValStrategy[1], k=k, replicates=10, # check names with 2 digits
                                                      importance=importance, ntree=ntree,
                                                      DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset,
                                                      crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))


  system.time(result$resp1Cat<- pipe_randomForest(df=dfCat, predInput=rev(predInputCat), responseVar=responseVarCat,
                                        shap=FALSE, repVi=repVi, # TODO shap with classification trees
                                        crossValStrategy=crossValStrategy[2], replicates=replicates,  # check names with 2 digits
                                        ntree=ntree, importance=importance, summarizePred=FALSE,
                                        DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset,
                                        crossValRatio=crossValRatio[1], RFmodel=RFmodel, verbose=verbose))


  tmp<- lapply(result, function(x) expect_s3_class(x, class="pipe_result.randomForest"))

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
    expect_s3_class(x$shap, class = "shapviz")
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
  predInputR<- raster::raster(nrows=3, ncols=5)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  # Put some NAs to detect rotations
  NAs<- expand.grid(col=1:ncol(predInputR), row=1:nrow(predInputR))
  NAs<- NAs[NAs$row > NAs$col, ]
  predInputR[NAs$row, NAs$col]<- NA

  resultR<- list()
  # predInput<- predInputR

  ## TODO: categorical variables for rasters
  df<- df[, names(predInputR)]

  suppressWarnings(future::plan(future::multicore))
  # DEBUG: future::plan(future::sequential, split=TRUE)
  filenameRasterPred<- paste0(tempdir(), "/testMap1.grd") # avoid overwrite
  resultR$summarizedPred<- pipe_randomForest(df, predInput=predInputR,
                                           repVi=repVi,
                                           crossValStrategy=crossValStrategy[1], k=k, replicates=replicates,
                                           importance=importance, ntree=ntree, summarizePred=TRUE,
                                           filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster,
                                           DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose)

  filenameRasterPred<- paste0(tempdir(), "/testMap2.grd") # avoid overwrite
  resultR$pred<- pipe_randomForest(df, predInput=predInputR[[rev(names(predInputR))]],
                             repVi=repVi,
                             crossValStrategy=crossValStrategy[2], replicates=replicates,
                             importance=importance, ntree=ntree, summarizePred=FALSE,
                             filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster,
                             DALEXexplainer=FALSE, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose)

  resultR$inMemory<- pipe_randomForest(df, predInput=predInputR,
                                                  repVi=repVi,
                                                  crossValStrategy=crossValStrategy[1], k=k, replicates=replicates,
                                                  importance=importance, ntree=ntree, summarizePred=TRUE,
                                                  tempdirRaster=tempdirRaster,
                                                  DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose)


  tmp<- lapply(resultR, function(x){
    expect_s4_class(x$predictions, class="Raster")
  })
  tmp<- expect_equal(names(resultR$summarizedPred$predictions), expected=c("mean", "sd", "se"))
  tmp<- expect_equal(names(resultR$pred$predictions), expected=paste0("Ozone_rep", 1:replicates))
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
  system.time(res<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, importance=importance,
                               ntree=ntree, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.randomForest")

  future::plan(future::multicore)
  system.time(res<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, importance=importance,
                               ntree=ntree, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.randomForest")

  future::plan(future.callr::callr(workers=3))
  system.time(res<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, importance=importance,
                               ntree=ntree, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.randomForest")

  future::plan(future::sequential)
  system.time(res<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, importance=importance,
                               ntree=ntree, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.randomForest")
})


test_that("scaleDataset", {
  future::plan(future::multisession)
  system.time(res<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi,
                               importance=importance, scaleDataset=TRUE, ntree=ntree,
                               DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))
  expect_s3_class(res, class="pipe_result.randomForest")

  predInputR<- raster::raster(nrows=5, ncols=5)
  predInputR<- raster::stack(lapply(varScale, function(i){
    raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
  }))

  # predInput<- predInputR

  ## TODO: categorical variables for rasters
  df<- df[, names(predInputR)]

  filenameRasterPred<- paste0(tempdir(), "/testMapScaleDataset.grd") # avoid overwrite
  res<- pipe_randomForest(df, predInput=predInputR, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi, importance=importance,
                   scaleDataset=TRUE,  ntree=ntree,
                   filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster,
                   DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose)
  expect_s3_class(res, class="pipe_result.randomForest")
})


test_that("summary", {
  future::plan(future::multisession)
  system.time(res<- pipe_randomForest(df=df, predInput=predInput, responseVar=responseVar, crossValStrategy=crossValStrategy[2], replicates=replicates, repVi=repVi,
                               importance=importance, scaleDataset=TRUE, ntree=ntree,
                               DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, RFmodel=RFmodel, verbose=verbose))

  sres<- summary(res)
  expect_s3_class(sres, class="summary.pipe_result.randomForest")
  expect_type(sres, type="list")
})

