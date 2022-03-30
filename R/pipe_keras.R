#' Neural network model with keras
#'
#' @param df a data.frame with response variable in the first column
#' @param predInput a Raster or a data.frame with columns 1 and 2 corresponding to longitude and latitude + variables for the model
#' @param responseVars response variables. Column names or indexes on df.
#' @param caseClass class of the samples used to weight cases. Column names or indexes on df, or a vector with the class for each rows in df.
#' @param idVars id column names or indexes on df and/or predInput. Deprecated default for compatibility c("x", "y")
#' @param weight Optional array of the same length as \code{nrow(df)}, containing weights to apply to the model's loss for each sample.
#' @param repVi replicates of the permutations to calculate the importance of the variables. 0 to avoid calculating variable importance.
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
pipe_keras<- function(df, predInput=NULL, responseVars=1, caseClass=NULL, idVars=character(), weight="class",
                   repVi=5, crossValStrategy=c("Kfold", "bootstrap"), k=5, replicates=10, crossValRatio=c(train=0.6, test=0.2, validate=0.2),
                   hidden_shape=50, epochs=500, maskNA=NULL, batch_size="all",
                   summarizePred=TRUE, scaleDataset=FALSE, NNmodel=FALSE, DALEXexplainer=FALSE, variableResponse=FALSE,
                   baseFilenameNN=NULL, filenameRasterPred=NULL, tempdirRaster=NULL, nCoresRaster=parallel::detectCores() %/% 2, verbose=0, ...){
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

  predVars<- setdiff(colnames(df), c(responseVars, idVars))
  predVars.cat<- names(which(!sapply(df[, predVars, drop=FALSE], is.numeric)))
  predVars.num<- setdiff(predVars, predVars.cat)

  if (length(predVars.cat) > 0){
    df.catBin<- stats::model.matrix(stats::as.formula(paste("~ -1 +", paste(predVars.cat, collapse="+"))), data=df)
    predVars.catBin<- colnames(df.catBin)
    df<- cbind(df[, setdiff(colnames(df), predVars.cat)], df.catBin)
    predVars<- c(predVars.num, predVars.catBin)
  }

  ## Select and sort predVars in predInput based on var names matching in df
  idVarsPred<- NULL
  if (!is.null(predInput)){
    if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
      ## TODO: onehot for categorical vars ----
      selCols<- intersect(predVars, names(predInput))
      predInput<- predInput[[selCols]]
      if (!identical(selCols, names(predInput)))
        stop("Input names for predictions doesn't match input for training. Check variable names.")
    } else if (inherits(predInput, c("data.frame", "matrix"))){
      if (length(predVars.cat) > 0){
        predInput.catBin<- stats::model.matrix(stats::as.formula(paste("~ -1 +", paste(predVars.cat, collapse="+"))), data=predInput)
        predInput<- cbind(predInput[, setdiff(colnames(predInput), predVars.cat)], predInput.catBin)
      }

      selCols<- intersect(predVars, colnames(predInput))
      idVarsPred<- intersect(idVars, colnames(predInput))

      if (length(idVarsPred) > 0){
        predInputIdVars<- predInput[, idVarsPred, drop=FALSE]
      }

      predInput<- as.matrix(predInput[, selCols])

      if (!identical(selCols, colnames(predInput)))
        stop("Input names for predictions doesn't match input for training. Check variable names.")
    }

  }

  if (scaleDataset){
    df.scaled<- scale(df[, predVars.num, drop=FALSE])
    df[, predVars.num]<- df.scaled
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

        # predInputScaled<- raster::scale(predInput[[names(col_means_train)]], center=col_means_train, scale=col_stddevs_train)
        # predInputScaled<- raster::calc(predInput[[names(col_means_train)]], filename=filenameScaled, fun=function(x) scale(x, center=col_means_train, scale=col_stddevs_train))
        raster::beginCluster(n=nCoresRaster)
        predInput[[names(col_means_train)]]<- raster::clusterR(predInput[[names(col_means_train)]], function(x, col_means_train, col_stddevs_train){
                              raster::calc(x, fun=function(y) scale(y, center=col_means_train, scale=col_stddevs_train))
                            }, args=list(col_means_train=col_means_train, col_stddevs_train=col_stddevs_train), filename=filenameScaled)
        if (!is.null(maskNA)){
          predInput<- raster::clusterR(predInput, function(x, maskNA){
                                raster::calc(x, fun=function(y) { y[is.na(y)]<- maskNA; y } )
                              }, args=list(maskNA=maskNA), filename=filenameScaled, overwrite=TRUE)
        }
        raster::endCluster()

      }  else if (inherits(predInput, c("data.frame", "matrix"))) {
        predInput[, predVars.num]<- scale(predInput[, predVars.num, drop=FALSE], center=col_means_train, scale=col_stddevs_train)
        if (!is.null(maskNA)){
          predInput<- apply(predInput, 2, function(x){
            x[is.na(x)]<- maskNA
            x
          })
        }
      }

    }
  }


  idxSetsL<- switch(crossValStrategy,
    bootstrap=bootstrap_train_test_validate(df, replicates=replicates, ratio=crossValRatio, caseClass=caseClass, weight=weight),
    Kfold=kFold_train_test_validate(d=df, k=k, caseClass=caseClass, weight=weight)
  )

  res<- future.apply::future_lapply(idxSetsL$replicates, function(idx.repli){
    resi<- list()
    # crossValSets<- splitdf(df, ratio=crossValRatio, sample_weight=sample_weight)
    crossValSets<- lapply(idx.repli[intersect(c("trainset", "testset"), names(idx.repli))], function(x) df[x, ])

    train_labels<- crossValSets$trainset[, responseVars, drop=FALSE]
    train_data<- crossValSets$trainset[, predVars, drop=FALSE]

    test_labels<- crossValSets$testset[, responseVars, drop=FALSE]
    test_data<- crossValSets$testset[, predVars, drop=FALSE]

    sample_weight<- idx.repli[intersect(c("weight.train", "weight.test"), names(idx.repli))]
    # if (length(sample_weight) == 0) sample_weight<- NULL

    # If no validation set exist, use test set to check performance
    if (length(idxSetsL$validateset) > 0){
      validate_labels<- df[idxSetsL$validateset, responseVars, drop=FALSE]
      validate_data<- df[idxSetsL$validateset, predVars, drop=FALSE]
      if (!is.null(idxSetsL$weight.validate)){
        sample_weight$weight.validate<- idxSetsL$weight.validate
      }
    } else {
      validate_labels<- test_labels
      validate_data<- test_data

      if (!is.null(idxSetsL$weight.validate)){
        sample_weight$weight.validate<- sample_weight$weight.test
      }
    }

    if (!scaleDataset){
      train_data.scaled<- scale(train_data[, predVars.num, drop=FALSE])
      train_data[, predVars.num]<- train_data.scaled

      col_means_train<- attr(train_data.scaled, "scaled:center")
      col_stddevs_train<- attr(train_data.scaled, "scaled:scale")
      rm(train_data.scaled)

      test_data[, predVars.num]<- scale(test_data[, predVars.num, drop=FALSE], center=col_means_train, scale=col_stddevs_train)
      validate_data[, predVars.num]<- scale(validate_data[, predVars.num, drop=FALSE], center=col_means_train, scale=col_stddevs_train)

      if (!is.null(maskNA)){
        train_data<- apply(train_data, 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
        test_data<- apply(test_data, 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
        validate_data<- apply(validate_data, 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
      }

      resi$scaleVals<- data.frame(mean=col_means_train, sd=col_stddevs_train)
    }

    if (inherits(train_data, "data.frame")){
      train_data<- as.matrix(train_data)
    }
    if (inherits(test_data, "data.frame")){
      test_data<- as.matrix(test_data)
    }
    if (inherits(validate_data, "data.frame")){
      validate_data<- as.matrix(validate_data)
    }

    if (inherits(train_labels, "data.frame")){
      train_labels<- as.matrix(train_labels)
    }
    if (inherits(test_labels, "data.frame")){
      test_labels<- as.matrix(test_labels)
    }
    if (inherits(validate_labels, "data.frame")){
      validate_labels<- as.matrix(validate_labels)
    }

    ## TODO: check if reset_state is faster and equivalent to build_model. Not possible to reuse model among replicates
    ## WARNING: Don't import/export NNmodel nor python objects to code inside future for PSOCK clusters, callR.
    # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
    modelNN<- build_modelDNN(input_shape=length(predVars), output_shape=length(responseVars), hidden_shape=hidden_shape, mask=maskNA)
    # modelNN<- keras::reset_states(modelNN)

    ## Check convergence on the max epochs frame
    early_stop<- keras::callback_early_stopping(monitor="val_loss", patience=30)

    modelNN<- train_keras(modelNN=modelNN, train_data=train_data, train_labels=train_labels,
                          test_data=test_data, test_labels=test_labels, epochs=epochs, batch_size=batch_size,
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
    resi$performance<- performance_keras(modelNN=modelNN, test_data=validate_data, test_labels=validate_labels,
                                         batch_size=ifelse(batch_size %in% "all", nrow(test_data), batch_size),
                                         sample_weight=sample_weight.validate, verbose=verbose)

    if (verbose > 1) message("Performance analyses done")

    ## Variable importance
    if (repVi > 0){
      variable_groups<- NULL
      if (length(predVars.cat) > 0){
        # Join variable importance for predictors from the same categorical variable
        variable_groups<- lapply(predVars.cat, function(x){
          grep(paste0("^", x), colnames(validate_data), value=TRUE)
        })
        variable_groups<- setNames(variable_groups, nm=predVars.cat)
        predVarsNumOri<- setdiff(predVars, predVars.catBin)
        variable_groups<- c(setNames(as.list(predVarsNumOri), nm=predVarsNumOri), variable_groups)
      }
      resi$variableImportance<- variableImportance_keras(model=modelNN, data=validate_data, y=validate_labels, repVi=repVi, variable_groups=variable_groups)
    }

    ## Explain model
    if (variableResponse | DALEXexplainer){
      explainer<- DALEX::explain(model=modelNN, data=validate_data, y=validate_labels, predict_function=stats::predict, label="MLP_keras", verbose=FALSE)

      ## Variable response
      if (variableResponse){
        resi$variableResponse<- variableResponse_keras(explainer)
      }

      if (DALEXexplainer){
        resi$explainer<- explainer
      }
    }
    if (verbose > 1) message("Model analyses done")

    ## Predictions
    if (!is.null(predInput)){
      if (inherits(predInput, "Raster")){
        batch_sizePred<- ifelse(batch_size %in% "all", raster::ncell(predInput), batch_size)
      } else {
        batch_sizePred<- ifelse(batch_size %in% "all", nrow(predInput), batch_size)
      }

      resi$predictions<- predict_keras(modelNN=modelNN, predInput=predInput, maskNA=maskNA,
                             scaleInput=!scaleDataset, col_means_train=col_means_train, col_stddevs_train=col_stddevs_train,
                             batch_size=batch_sizePred, tempdirRaster=tempdirRaster, nCoresRaster=nCoresRaster)
      if (inherits(resi$predictions, "matrix")){
        colnames(resi$predictions)<- responseVars
        rownames(resi$predictions)<- rownames(predInput)
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
