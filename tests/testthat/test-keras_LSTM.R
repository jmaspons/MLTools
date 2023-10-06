context("LSTM_keras")

df<- data.frame(id=rep(LETTERS[1:10], each=5), static=rep(1:10, each=5), time=rep(1:5, times=5))
df.cat<- data.frame(id=LETTERS[1:10], cat1=rep(LETTERS[1:5], times=2), cat2=letters[1:10])
df<- merge(df, df.cat)
df$x1<- df$time * df$static
df$x2<- rnorm(nrow(df), mean=df$time * df$static + 100, sd=5)
df$x3<- rnorm(nrow(df), mean=df$time * df$static * 3, sd=2)
df$y<- rnorm(nrow(df), mean=(df$x1 + df$x2) / df$x3, sd=2)

timevar<- "time"
idVars<- "id"
responseVars<- "y"
staticVars<- c("static", "cat1", "cat2")
predTemp<- c("x1", "x2", "x3")
predInput<- df


predInput<- predInput
responseVars<- responseVars
caseClass<- NULL
idVars<- idVars
weight<- "class"
modelType<- "LSTM"
timevar<- timevar
responseTime<- "LAST"
regex_time<- "[0-9]+"
staticVars<- staticVars
repVi<- 2
perm_dim<- list(2:3, 2)
comb_dims<- FALSE
crossValStrategy<- c("Kfold", "bootstrap")
k<- 3
replicates<- 2
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
hidden_shape.RNN<- 2
hidden_shape.static<- 2
hidden_shape.main<- 3
epochs<- 2
maskNA<- -999
batch_size<- 1000
summarizePred<- TRUE
scaleDataset<- FALSE
NNmodel<- TRUE
DALEXexplainer<- TRUE
variableResponse<- TRUE
save_validateset<- TRUE
baseFilenameNN<- NULL
filenameRasterPred<- NULL
tempdirRaster<- NULL
nCoresRaster<- parallel::detectCores() %/% 2
verbose<- 0

## TODO: FIX IT ----
variableResponse=FALSE


test_that("keras_LSTM works", {
  result<- list()

  # future::plan(future::sequential, split=TRUE)
  future::plan(future::multisession)
  # future::futureSessionInfo()
  system.time(result$resp1summarizedPred<- pipe_keras_timeseries(df=df, predInput=predInput, responseVars=responseVars, caseClass=caseClass, idVars=idVars, weight=weight,
                                                                 timevar=timevar, responseTime=responseTime, regex_time=regex_time, staticVars=staticVars,
                                                                 repVi=repVi, perm_dim=perm_dim, comb_dims=comb_dims, crossValStrategy=crossValStrategy[1], k=k, replicates=replicates, crossValRatio=crossValRatio,
                                                                 hidden_shape.RNN=c(2, 2), hidden_shape.static=c(2, 2), hidden_shape.main=c(2, 2), epochs=epochs, maskNA=maskNA, batch_size=batch_size,
                                                                 summarizePred=TRUE, scaleDataset=scaleDataset, NNmodel=NNmodel, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset)
  )

  system.time(result$resp2summarizedPred<- pipe_keras_timeseries(df=df, predInput=predInput, responseVars=c("y", "x1"), caseClass=caseClass, idVars=idVars, weight=weight,
                                                                 timevar=timevar, responseTime=responseTime, regex_time=regex_time, staticVars=staticVars,
                                                                 repVi=repVi, perm_dim=perm_dim, comb_dims=TRUE, crossValStrategy=crossValStrategy[2], replicates=replicates, crossValRatio=crossValRatio,
                                                                 hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, epochs=epochs, maskNA=maskNA, batch_size=batch_size,
                                                                 summarizePred=TRUE, scaleDataset=scaleDataset, NNmodel=NNmodel, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset)
  )

  system.time(result$resp1<- pipe_keras_timeseries(df=df, predInput=predInput, responseVars=responseVars, caseClass=caseClass, idVars=idVars, weight=weight,
                                                   timevar=timevar, responseTime=responseTime, regex_time=regex_time, staticVars=staticVars,
                                                   repVi=10, perm_dim=perm_dim, comb_dims=TRUE, crossValStrategy=crossValStrategy[2], replicates=10, crossValRatio=crossValRatio, # check names with 2 digits (replicates & repVi = 10)
                                                   hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, epochs=epochs, maskNA=maskNA, batch_size=batch_size,
                                                   summarizePred=FALSE, scaleDataset=scaleDataset, NNmodel=NNmodel, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset)
  )

  system.time(result$resp2<- pipe_keras_timeseries(df=df, predInput=predInput, responseVars=responseVars, caseClass=caseClass, idVars=idVars, weight=weight,
                                                   timevar=timevar, responseTime=responseTime, regex_time=regex_time, staticVars=staticVars,
                                                   repVi=repVi, perm_dim=perm_dim, comb_dims=comb_dims, crossValStrategy=crossValStrategy[1], k=k, replicates=replicates, crossValRatio=crossValRatio,
                                                   hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, epochs=epochs, maskNA=maskNA, batch_size=batch_size,
                                                   summarizePred=FALSE, scaleDataset=scaleDataset, NNmodel=NNmodel, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse, save_validateset=save_validateset)
  )

  tmp<- lapply(result, function(x) expect_s3_class(x, class="pipe_result.keras"))

  tmp<- lapply(result, function(x){
      expect_s3_class(x$performance, class="data.frame")
      reps<- nrow(x$performance)
      if (x$params$crossValStrategy == "bootstrap"){
        expect_equal(rownames(x$performance), expected=paste0("rep", formatC(1:reps, format="d", flag="0", width=nchar(reps))))
      } else if (x$params$crossValStrategy == "Kfold") {
        expect_equal(rownames(x$performance), expected=paste0("Fold", 2:k, ".Rep", rep(1:replicates, each=k-1)))  # Fold2:k (Fold1 for validationset)
      }
    })

  tmp<- lapply(result, function(x){
    expect_type(x$scale, type="list")
    expect_equal(unique(lapply(x$scale, names)), expected=list(c("mean", "sd")))
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

  ## TODO: variableResponse not implemented for 3d arrays yet. Fix ingredients::partial_dependence ----
  # tmp<- lapply(result, function(x){
  #   expect_type(x$variableCoef, type="list")
  #   lapply(x$variableCoef, function(y){
  #     expectedColnames<- c("intercept", paste0("b", 1:(ncol(y) - 4)), "adj.r.squared", "r.squared", "degree")
  #     expect_equal(colnames(y), expected=expectedColnames)
  #   })
  # })

  tmp<- lapply(result, function(x){
    expect_type(x$predictions, type="list")
    sapply(x$predictions, expect_s3_class, class="data.frame")
  })
  expectedColnames<- c(idVars, "Mean", "SD", "Naive SE", "2.5%", "25%", "50%", "75%", "97.5%")
  tmp<- expect_equal(colnames(result$resp1summarizedPred$predictions[[1]]), expected=expectedColnames)
  tmp<- expect_equal(unlist(unique(lapply(result$resp2summarizedPred$predictions, colnames))), expected=expectedColnames)
  tmp<- expect_equal(colnames(result$resp1$predictions[[1]]), expected=c(idVars, paste0("rep", formatC(1:nrow(result$resp1$performance), format="d", flag="0", width=nchar(nrow(result$resp1$performance))))))
  tmp<- expect_equal(unlist(unique(lapply(result$resp2$predictions, colnames))), expected=c(idVars, paste0("rep", formatC(1:nrow(result$resp2$performance), format="d", flag="0", width=nchar(nrow(result$resp2$performance))))))

  tmp<- lapply(result, function(x){
    expect_type(x$model, type="list")
    lapply(x$model, function(y) expect_type(y, type="raw"))
    lapply(x$model, function(y) expect_s3_class(keras::unserialize_model(y), class="keras.engine.training.Model"))
  })
  # dir(tempdir(), full.names=TRUE)
  # expect_true(any(grepl(baseFilenameNN, dir(tempdir(), full.names=TRUE))))
  # expect_equal(sum(grepl(baseFilenameNN, dir(tempdir(), full.names=TRUE))), k - 1)
## TODO: DALEXexplainer not implemented for multiinput models
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


# test_that("keras_LSTM predict with raster", {
#   predInputR<- raster::raster(nrows=4, ncols=6)
#   predInputR<- raster::stack(lapply(varScale, function(i){
#     raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
#   }))
#
#   # Put some NAs to detect rotations
#   NAs<- expand.grid(col=1:ncol(predInputR), row=1:nrow(predInputR))
#   NAs<- NAs[NAs$row > NAs$col, ]
#   predInputR[NAs$row, NAs$col]<- NA
#
#   names(predInputR)<- names(varScale)
#   resultR<- list()
#   # predInput<- predInputR
#
#
#   suppressWarnings(future::plan(future::multicore))
#   filenameRasterPred<- paste0(tempdir(), "/testMap1.grd") # avoid overwrite
#   resultR$resp1summarizedPred<- pipe_keras(df, predInput=predInputR,
#                                               epochs=epochs, repVi=repVi,
#                                               crossValStrategy=crossValStrategy[1], k=k,
#                                               batch_size=batch_size, hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, summarizePred=TRUE,
#                                               filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
#                                               DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
#
#   filenameRasterPred<- paste0(tempdir(), "/testMap2.grd") # avoid overwrite
#   resultR$resp1<- pipe_keras(df, predInput=predInputR[[rev(names(predInputR))]],
#                                 epochs=epochs, maskNA=maskNA, repVi=repVi,
#                                 crossValStrategy=crossValStrategy[2], replicates=replicates,
#                                 batch_size=batch_size, hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, summarizePred=FALSE,
#                                 filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
#                                 DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
#
#   filenameRasterPred<- paste0(tempdir(), "/testMap3.grd") # avoid overwrite
#   resultR$resp2summarizedPred<- pipe_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, maskNA=maskNA, repVi=repVi,
#                         crossValStrategy=crossValStrategy[1], k=k, batch_size=batch_size, hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main,
#                         summarizePred=TRUE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
#                         DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
#
#   filenameRasterPred<- paste0(tempdir(), "/testMap4.grd") # avoid overwrite
#   resultR$resp2<- pipe_keras(df, predInput=predInputR, responseVars=1:2, epochs=epochs, repVi=repVi, replicates=replicates, batch_size=batch_size, hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main,
#                          summarizePred=FALSE, filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
#                          DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
#
#   tmp<- lapply(resultR, function(x){
#     expect_s4_class(x$predictions, class="RasterBrick")
#   })
#   tmp<- expect_equal(names(resultR$resp1summarizedPred$predictions), expected=c("mean", "sd", "se"))
#   tmp<- expect_equal(names(resultR$resp1$predictions), expected=paste0("X1_rep", 1:replicates))
#
#   # lapply(resultR, function(x) names(x$predictions))
#   ## Check NAs position
#   # plot(predInputR)
#   # plot(resultR$resp1summarizedPred$predictions)
#   # plot(resultR$resp1$predictions)
#   # plot(resultR$resp2summarizedPred$predictions)
#   # plot(resultR$resp2$predictions)
#
#   file.remove(dir(tempdir(), "testMap.+\\.gr(i|d)$", full.names=TRUE))
# })
#
#
# test_that("Future plans work with keras_LSTM", {
#   # options(future.globals.onReference = "error")
# # Error in keras::reset_states(modelNN) : attempt to apply non-function
# # Don't import/export python objects to/from code inside future for PSOCK and callR clusters
# # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
#
#   suppressWarnings(future::plan(future::transparent))
#   system.time(res<- pipe_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
#                                   hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
#   expect_s3_class(res, class="pipe_result.keras")
#
#   future::plan(future::multicore)
#   system.time(res<- pipe_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
#                                   hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
#   expect_s3_class(res, class="pipe_result.keras")
#
#   future::plan(future.callr::callr(workers=3))
#   system.time(res<- pipe_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
#                                   hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
#   expect_s3_class(res, class="pipe_result.keras")
#
#   future::plan(future::sequential)
#   system.time(res<- pipe_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
#                                   hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
#   expect_s3_class(res, class="pipe_result.keras")
# })


# test_that("scaleDataset with keras_LSTM", {
#   future::plan(future::multisession)
#   system.time(res<- pipe_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
#                                    batch_size=batch_size, scaleDataset=TRUE, hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main,
#                                    baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
#   expect_s3_class(res, class="pipe_result.keras")
#
#   predInputR<- raster::raster(nrows=15, ncols=15)
#   predInputR<- raster::stack(lapply(varScale, function(i){
#     raster::setValues(predInputR, runif(raster::ncell(predInputR)) * i)
#   }))
#
#   names(predInputR)<- names(df)
#   # predInput<- predInputR
#
#   filenameRasterPred<- paste0(tempdir(), "/testMapScaleDataset.grd") # avoid overwrite
#   res<- pipe_keras(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
#                         scaleDataset=TRUE,  hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main,
#                         filenameRasterPred=filenameRasterPred, tempdirRaster=tempdirRaster, baseFilenameNN=baseFilenameNN,
#                         DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
#   expect_s3_class(res, class="pipe_result.keras")
# })


# test_that("keras_LSTM summary", {
#   future::plan(future::multisession)
#   system.time(res<- pipe_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi,
#                                    batch_size=batch_size, scaleDataset=TRUE, hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main,
#                                    baseFilenameNN=baseFilenameNN, DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
#
#   sres<- summary(res)
#   expect_s3_class(sres, class="summary.pipe_result.keras")
#   expect_type(sres, type="list")
# })

