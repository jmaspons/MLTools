#' Neural network model with keras
#'
#' @param df a data.frame with response variable in the first column
#' @param predInput a Raster or a data.frame with columns 1 and 2 corresponding to longitude and latitude + variables for the model
#' @param responseVars response variables. Column names or indexes on df in wide format (eg. respVar_time).
#' @param caseClass class of the samples used to weight cases. Column names or indexes on df, or a vector with the class for each rows in df.
#' @param idVars id column names or indexes on df and/or predInput. Should be a unique identifier for a row in wide format. Deprecated default for compatibility c("x", "y")
#' @param weight Optional array of the same length as \code{nrow(df)}, containing weights to apply to the model's loss for each sample.
#' @param modelType type of neural network. "DNN" for Deep Neural Network, "LSTM" for Long Short-Term Memory (\code{input_data} with time-series in long format and a column for time. \code{timevar} parameter needed).
#' @param timevar column name of the variable containing the time. Use with modelType = "LSTM".
#' @param responseTime a \code{timevar} value used as a response var for \code{responseVars} or the default "LAST" for the last timestep available (\code{max(df[, timevar])}).
#' @param regex_time regular expression matching the \code{timevar} values format.
#' @param repVi replicates of the permutations to calculate the importance of the variables. 0 to avoid report variable importance
#' @param crossValStrategy \code{Kfold} or \code{bootstrap}.
#' @param replicates number of replicates when \code{crossValStrategy="bootstrap"}.
#' @param k number of data partitions when \code{crossValStrategy="Kfold"}.
#' @param crossValRatio Proportion of the dataset used to train, test and validate the model when \code{crossValStrategy="bootstrap"}. Default to \code{c(train=0.6, test=0.2, validate=0.2)}. If there is only one value, will be taken as a train proportion and no validate set.
#' @param hidden_shape number of neurons in the hidden layers of the neural network model.
#' @param epochs parameter for \code\link[keras]{fit}}.
#' @param batch_size for fit and predict functions. The bigger the better if it fits your available memory. Integer or "all".
#' @param summarizePred if \code{TRUE}, return the mean, sd and se of the predictors. if \code{FALSE}, return the predictions for each replicate.
#' @param scaleDataset if \code{TRUE}, scale the whole dataset only once instead of the train set at each replicate. Optimize processing time for predictions with large rasters.
#' @param NNmodel if TRUE, return the serialized model with the result.
#' @param DALEXexplainer if TRUE, return a explainer for the models from \code\link[DALEX]{explain}} function. It doesn't work with multisession future plans.
#' @param variableResponse if TRUE, return aggregated_profiles_explainer object from \code\link[ingredients]{partial_dependency}} and the coefficients of the adjusted lineal model.
#' @param baseFilenameNN if no missing, save the NN in hdf5 format on this path with iteration appended.
#' @param filenameRasterPred if no missing, save the predictions in a RasterBrick to this file.
#' @param tempdirRaster path to a directory to save temporal raster files.
#' @param nCoresRaster number of cores used for parallelized raster cores. Use half of the available cores by default.
#' @param verbose If > 0, print state and passed to keras functions
#' @param ... extra parameters for \code\link[future.apply]{future_replicate}}. Better that user use future::plan?
#'
#' @return
#' @export
#' @import keras
#' @importFrom stats predict
#' @examples
pipe_keras_timeseries<- function(df, predInput=NULL, responseVars=1, caseClass=NULL, idVars=character(), weight="class",
                   modelType=c("DNN", "LSTM"), timevar=NULL, responseTime="LAST", regex_time=".+", staticVars=NULL,
                   repVi=5, crossValStrategy=c("Kfold", "bootstrap"), k=5, replicates=10, crossValRatio=c(train=0.6, test=0.2, validate=0.2),
                   hidden_shape=50, epochs=500, maskNA=NULL, batch_size="all",
                   summarizePred=TRUE, scaleDataset=FALSE, NNmodel=FALSE, DALEXexplainer=FALSE, variableResponse=FALSE,
                   baseFilenameNN=NULL, filenameRasterPred=NULL, tempdirRaster=NULL, nCoresRaster=parallel::detectCores() %/% 2, verbose=0, ...){
  modelType<- match.arg(modelType)
  crossValStrategy<- match.arg(crossValStrategy)
  if (is.numeric(responseVars)){
    responseVars<- colnames(df)[responseVars]
  }
  if (is.numeric(idVars)){
    idVars<- colnames(df)[idVars]
  }
  if (length(caseClass) == 1){
    if (is.numeric(caseClass)){
      idVars<- c(idVars, colnames(df)[caseClass])
    } else if (is.character(caseClass)){
      idVars<- c(idVars, caseClass)
    }
  }
  if (responseTime == "LAST"){
    responseTime<- max(df[, timevar], na.rm=TRUE)
  }

  predVars<- setdiff(colnames(df), c(idVars, timevar))

  ## Select and sort predVars in predInput based on var names matching in df
  idVarsPred<- NULL
  if (!is.null(predInput)){
    if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
      selCols<- intersect(predVars, names(predInput))
      predInput<- predInput[[selCols]]
      if (!identical(selCols, names(predInput)))
        stop("Input names for predictions doesn't match input for training. Check variable names.")
    }  else if (inherits(predInput, c("data.frame", "matrix"))) {
      selCols<- intersect(predVars, colnames(predInput))
      idVarsPred<- intersect(idVars, colnames(predInput))

      if (length(idVarsPred) > 0){
        predInputIdVars<- predInput[, idVarsPred, drop=FALSE]
      }

      predInput<- predInput[, selCols]

      if (!identical(selCols, colnames(predInput)))
        stop("Input names for predictions doesn't match input for training. Check variable names.")
    }

  }

  if (scaleDataset){
    df.scaled<- scale(df[, predVars, drop=FALSE])
    df[, predVars]<- df.scaled
    col_means_train<- attr(df.scaled, "scaled:center")
    col_stddevs_train<- attr(df.scaled, "scaled:scale")
    rm(df.scaled)

    if (!is.null(maskNA)){
      df[, predVars]<- apply(df[, predVars, drop=FALSE], 2, function(x){
        x[is.na(x)]<- maskNA
        x
      })
    }

    if (!is.null(predInput)){

      if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
        if (!is.null(tempdirRaster)){
          filenameScaled<- tempfile(tmpdir=tempdirRaster, fileext=".grd")
        } else {
          filenameScaled<- raster::rasterTmpFile()
        }

        # predInputScaled<- raster::scale(predInput, center=col_means_train, scale=col_stddevs_train)
        # predInputScaled<- raster::calc(predInput, filename=filenameScaled, fun=function(x) scale(x, center=col_means_train, scale=col_stddevs_train))
        raster::beginCluster(n=nCoresRaster)
        predInput<- raster::clusterR(predInput, function(x, col_means_train, col_stddevs_train){
                              raster::calc(x, fun=function(y) scale(y, center=col_means_train, scale=col_stddevs_train))
                            }, args=list(col_means_train=col_means_train, col_stddevs_train=col_stddevs_train), filename=filenameScaled)
        if (!is.null(maskNA)){
          predInput<- raster::clusterR(predInput, function(x, maskNA){
                                raster::calc(x, fun=function(y) { y[is.na(y)]<- maskNA; y } )
                              }, args=list(maskNA=maskNA), filename=filenameScaled, overwrite=TRUE)
        }
        raster::endCluster()

      }  else if (inherits(predInput, c("data.frame", "matrix"))) {
        predInput[, predVars]<- scale(predInput[, predVars, drop=FALSE], center=col_means_train, scale=col_stddevs_train)
        if (!is.null(maskNA)){
          predInput[, predVars]<- apply(predInput[, predVars, drop=FALSE], 2, function(x){
            x[is.na(x)]<- maskNA
            x
          })
        }
      }

    }
  }

  # crossvalidation for timeseries must be done in the wide format data
  # scaling must be done on long format data
  ## NEEDS a refactor :_( or changing data formats twice or map cases from wide to long
  ## TODO: QUESTION: How to build de 3Darray? Discard responseTime VS responseTime for responseVars <- NA VS irregular shape coder/encoder layer ----
  # if (modelType == "LSTM"){
  responseVars.ts<- paste0(responseVars, "_", responseTime)
  predVars.tf<- paste0(setdiff(predVars, staticVars), "_", responseTime)
  df.long<- df
  df.wide<- longToWide.ts(d=df, timevar=timevar, idCols=c(idVars, staticVars))
  # vars<- colnames(df.long[, predVars])
  predVars.ts<- setdiff(colnames(df.wide), c(idVars, staticVars)) # WARNING: Includes responseVars.ts
  timevals<- unique(df[[timevar]])

  # }


  idxSetsL<- switch(crossValStrategy,
    bootstrap=bootstrap_train_test_validate(df.wide, replicates=replicates, ratio=crossValRatio, caseClass=caseClass, weight=weight),
    Kfold=kFold_train_test_validate(d=df.wide, k=k, caseClass=caseClass, weight=weight)
  )

  res<- future.apply::future_lapply(idxSetsL$replicates, function(idx.repli){
    # TODO: TEST idx.repli<- idxSetsL$replicates[[1]]
    resi<- list()
    crossValSets<- lapply(idx.repli[intersect(c("trainset", "testset"), names(idx.repli))], function(x) df.wide[x, ])

    sample_weight<- idx.repli[intersect(c("weight.train", "weight.test"), names(idx.repli))]

    # If no validation set exist, use test set to check performance
    if (length(idxSetsL$validateset) > 0){
      validationSet<- df.wide[idxSetsL$validateset, ]
      if (!is.null(idxSetsL$weight.validate)){
        sample_weight$weight.validate<- idxSetsL$weight.validate
      }
    } else {
      validationSet<- crossValSets$testset
      if (!is.null(idxSetsL$weight.validate)){
        sample_weight$weight.validate<- sample_weight$weight.test
      }
    }

    if (!scaleDataset){
      # NOTE: if *_data are matrix, wideToLong.ts transform timevar to character, all columns become characters.
      # Use data.frames with unique idVars cases will be aggregated when wideToLong.ts ----
      trainset.long<- wideToLong.ts(d=crossValSets$trainset, timevar=timevar, vars=setdiff(predVars, c(idVars, staticVars)), idCols=c(idVars, staticVars), regex_time=regex_time)
      trainset.longScaled<- scale(trainset.long[, predVars])
      trainset.long[, predVars]<- trainset.longScaled
      col_means_train<- attr(trainset.longScaled, "scaled:center")
      col_stddevs_train<- attr(trainset.longScaled, "scaled:scale")
      ## NOTE: must contain idVars and idVars should be a unique id for each case in a wide format ----
      trainset<- longToWide.ts(d=trainset.long, timevar=timevar, idCols=c(idVars, staticVars))

      # Scale trestset in wide format
      matchVars<- data.frame(predVars=gsub(paste0("_(", paste(timevals, collapse="|"), ")$"), "", predVars.ts),
                             predVars.ts, stringsAsFactors=FALSE)
      matchVars<- rbind(matchVars, data.frame(predVars=staticVars, predVars.ts=staticVars))
      matchVars<- merge(data.frame(predVars=names(col_means_train), col_means_train=col_means_train, col_stddevs_train=col_stddevs_train),
                        matchVars, by="predVars")
      # rownames(matchVars)<- matchVars$predVars.ts
      # matchVars<- na.omit(matchVars)
      crossValSets$testset[, matchVars$predVars.ts]<- scale(crossValSets$testset[, matchVars$predVars.ts],
                                                            center=matchVars[, "col_means_train"],
                                                            scale=matchVars[, "col_stddevs_train"])
      validationSet[, matchVars$predVars.ts]<- scale(validationSet[, matchVars$predVars.ts],
                                                     center=matchVars[, "col_means_train"],
                                                     scale=matchVars[, "col_stddevs_train"])

      if (!is.null(maskNA)){
        selCols<- setdiff(colnames(crossValSets$trainset), idVars)
        crossValSets$trainset[, selCols]<- apply(crossValSets$trainset[, selCols], 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
        crossValSets$testset[, selCols]<- apply(crossValSets$testset[, selCols], 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
        validationSet[, selCols]<- apply(validationSet[, selCols], 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
      }

      if (!is.null(predInput)){
        predInput[, predVars]<- scale(predInput[, predVars, drop=FALSE], center=col_means_train, scale=col_stddevs_train)
        if (!is.null(maskNA)){
          predInput[, predVars]<- apply(predInput[, predVars, drop=FALSE], 2, function(x){
            x[is.na(x)]<- maskNA
            x
          })
        }
      }

      resi$scaleVals<- data.frame(mean=col_means_train, sd=col_stddevs_train)
    }

    train_labels<- crossValSets$trainset[, c(idVars, responseVars.ts), drop=FALSE]
    train_data<- crossValSets$trainset[, c(idVars, staticVars, predVars.ts), drop=FALSE]

    test_labels<- crossValSets$testset[, c(idVars, responseVars.ts), drop=FALSE]
    test_data<- crossValSets$testset[, c(idVars, staticVars, predVars.ts), drop=FALSE]

    if (length(idxSetsL$validateset) > 0){
      validate_labels<- validationSet[, c(idVars, responseVars.ts), drop=FALSE]
      validate_data<- validationSet[, c(idVars, staticVars, predVars.ts), drop=FALSE]
      if (!is.null(idxSetsL$weight.validate)){
        ## TODO: check case weights match data set cases ----
        sample_weight$weight.validate<- idxSetsL$weight.validate
      }
    } else {
      validate_labels<- test_labels
      validate_data<- test_data

      if (!is.null(idxSetsL$weight.validate)){
        ## TODO: check case weights match data set cases ----
        sample_weight$weight.validate<- sample_weight$weight.test
      }
    }


    # Reshape data to 3D arrays [samples, timesteps, features] as expected by LSTM layer
    # TODO: layer for static vars
    train_data.3d<- wideTo3Darray.ts(d=train_data, vars=setdiff(predVars, staticVars), idCols=idVars)
    test_data.3d<- wideTo3Darray.ts(d=test_data, vars=setdiff(predVars, staticVars), idCols=idVars)
    validate_data.3d<- wideTo3Darray.ts(d=validate_data, vars=setdiff(predVars, staticVars), idCols=idVars)

    ## TODO decide if responseVars.ts<- NA or remove predVars.tf ----
    if (is.null(maskNA)){
      train_data.3d<- train_data.3d[, setdiff(dimnames(train_data.3d)[[2]], responseTime), ]
    } else {
    # VS
      train_data.3d[, as.character(responseTime), responseVars]<- maskNA
    }
    ## TODO: check if reset_state is faster and equivalent to build_model. Not possible to reuse model among replicates
    ## WARNING: Don't import/export NNmodel nor python objects to code inside future for PSOCK clusters, callR.
    # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
    # modelNN<- keras::reset_states(modelNN)

    modelNN<- NNTools:::build_modelLTSM(input_shape=dim(train_data.3d)[-1], output_shape=length(responseVars),
                                          hidden_shape=hidden_shape, mask=maskNA)

    ## Check convergence on the max epochs frame
    early_stop<- keras::callback_early_stopping(monitor="val_loss", patience=30)

    # dim(train_data.3d)
    # dim(train_labels[, responseVars.ts, drop=FALSE])
    # dim(trainset.long)
    # dim(train_data)
    # dimnames(train_data.3d)
    # dimnames(train_labels[, responseVars.ts, drop=FALSE])

    ## NOTE: all data must be matrix or array or a list of if the model has multiple inputs
    modelNN<- NNTools:::train_keras(modelNN=modelNN, train_data=train_data.3d, train_labels=as.matrix(train_labels[, responseVars.ts, drop=FALSE]),
                           test_data=test_data.3d, test_labels=as.matrix(test_labels[, responseVars.ts, drop=FALSE]), epochs=epochs, batch_size=batch_size,
                           sample_weight=sample_weight, callbacks=early_stop, verbose=verbose)

    if (verbose > 1) message("Training done")

    if (NNmodel){
      resi$model<- keras::serialize_model(modelNN)
    }

    ## Model performance
    if (length(sample_weight) > 0){
      sample_weight.validate<- as.matrix(sample_weight$weight.validate)
    } else {
      sample_weight.validate<- NULL
    }
    resi$performance<- NNTools:::performance_keras(modelNN=modelNN, test_data=validate_data.3d, test_labels=as.matrix(validate_labels[, responseVars.ts, drop=FALSE]),
                             batch_size=ifelse(batch_size %in% "all", nrow(test_data), batch_size),
                             sample_weight=sample_weight.validate, verbose=verbose)

    if (verbose > 1) message("Performance analyses done")

    ## Explain model
    ## TODO: fix Explain model incorrect number of dimensions. Perhaps not ready for 3d data? ----
    if (repVi > 0 | variableResponse | DALEXexplainer){
      explainer<- DALEX::explain(model=modelNN, data=validate_data.3d, y=as.matrix(validate_labels[, responseVars.ts, drop=FALSE]), predict_function=stats::predict, label="MLP_keras", verbose=FALSE)

      ## Variable importance
      if (repVi > 0){
        resi$variableImportance<- NNTools:::variableImportance_keras(explainer=explainer, repVi=repVi)
      }
      ## Variable response
      if (variableResponse){
        resi$variableResponse<- variableResponse_keras(explainer)
      }

      if (DALEXexplainer){
        resi$explainer<- explainer
      }
      if (verbose > 1) message("Model analyses done")
    }


    ## Predictions
    if (!is.null(predInput)){
      if (inherits(predInput, "Raster")){
        batch_sizePred<- ifelse(batch_size %in% "all", raster::ncell(predInput), batch_size)
      } else {
        batch_sizePred<- ifelse(batch_size %in% "all", nrow(predInput), batch_size)
        predInput.3d<- longTo3Darray.ts(d=predInput, timevar=timevar, idCols=c(idVars, staticVars))
      }

      resi$predictions<- NNTools:::predict_keras(modelNN=modelNN, predInput=predInput.3d,
                               scaleInput=FALSE, batch_size=batch_sizePred, tempdirRaster=tempdirRaster, nCoresRaster=nCoresRaster)
      if (inherits(resi$predictions, "matrix")){
        colnames(resi$predictions)<- responseVars
        rownames(resi$predictions)<- dimnames(predInput.3d)$case
      } else if (inherits(resi$predictions, "Raster")){
        names(resi$predictions)<- responseVars
      }
    }
    if (verbose > 1) message("Prediction done")

    return(resi)
  }, future.seed=TRUE, ...)

  if (scaleDataset){
    res[[1]]$scaleVals<- list(dataset=data.frame(mean=col_means_train, sd=col_stddevs_train))
    if (inherits(predInput, "Raster")){
      file.remove(filenameScaled, gsub("\\.grd$", ".gri", filenameScaled))
    }
  }

  if (verbose > 0) message("Iterations finished. Gathering results...")

  out<- gatherResults.process_NN(res=res, summarizePred=summarizePred, filenameRasterPred=filenameRasterPred, nCoresRaster=nCoresRaster)

  if (!is.null(predInput) & inherits(predInput, c("data.frame", "matrix")) & length(idVarsPred) > 0){
    ## Add idVars if exists
    out$predictions<- lapply(out$predictions, function(x){
      cbind(predInputIdVars, x)
    })
  }

  if (!is.null(baseFilenameNN) & !is.null(res[[1]]$model)){
    tmp<- lapply(seq_along(out$model), function(i){
      save_model_hdf5(keras::unserialize_model(out$model[[i]]), filepath=paste0(baseFilenameNN, "_", formatC(i, format="d", flag="0", width=nchar(length(res))), ".hdf5"),
                      overwrite=TRUE, include_optimizer=TRUE)
    })
  }

  class(out)<- "process_NN"

  return(out)
}
