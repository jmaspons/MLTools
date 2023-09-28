#' Neural network model with keras
#'
#' @param df a `data.frame` with the data in a long format (time variable in the `timevar` column).
#' @param predInput a `data.frame` with the input variables to make predictions. The columns names must match the names of `df` columns.
#' @param responseVars response variables as column names or indexes on `df` in wide format (eg. respVar_time).
#' @param caseClass class of the samples used to weight cases. Column names or indexes on `df`, or a vector with the class for each rows in `df`.
#' @param idVars id column names or indexes on `df`. Should be a unique identifier for a row in wide format, otherwise, values will be averaged.
#' @param weight Optional array of the same length as `nrow(df)`, containing weights to apply to the model's loss for each sample.
#' @param timevar column name of the variable containing the time.
#' @param responseTime a `timevar` value used as a response var for `responseVars` or the default "LAST" for the last timestep available (`max(df[, timevar])`).
#' @param regex_time regular expression matching the `timevar` values format.
#' @param repVi replicates of the permutations to calculate the importance of the variables. 0 to avoid calculating variable importance.
#' @param perm_dim dimension to perform the permutations to calculate the importance of the variables (data dimensions \[case, time, variable\]).
#' If `perm_dim = 2:3`, it calculates the importance for each combination of the 2nd and 3rd dimensions.
#' @param comb_dims variable importance calculations, if `TRUE`, do the permutations for each combination of the levels of the variables from 2nd and 3rd dimensions for input data with 3 dimensions. By default `FALSE`.
#' @param crossValStrategy `Kfold` or `bootstrap`.
#' @param replicates number of replicates for `crossValStrategy="bootstrap"` and `crossValStrategy="Kfold"` (`replicates * k-1`, 1 fold for validation).
#' @param k number of data partitions when `crossValStrategy="Kfold"`.
#' @param crossValRatio Proportion of the dataset used to train, test and validate the model when `crossValStrategy="bootstrap"`. Default to `c(train=0.6, test=0.2, validate=0.2)`. If there is only one value, will be taken as a train proportion and the test set will be used for validation.
#' @param hidden_shape.RNN number of neurons in the hidden layers of the Recursive Neural Network model (time series data). Can be a vector with values for each hidden layer.
#' @param hidden_shape.static number of neurons in the hidden layers of the densely connected neural network model (static data). Can be a vector with values for each hidden layer.
#' @param hidden_shape.main number of neurons in the hidden layers of the densely connected neural network model connecting static and time series data. Can be a vector with values for each hidden layer.
#' @param epochs parameter for \code\link[keras]{fit}}.
#' @param batch_size for fit and predict functions. The bigger the better if it fits your available memory. Integer or "all".
#' @param summarizePred if `TRUE`, return the mean, sd and se of the predictors. if `FALSE`, return the predictions for each replicate.
#' @param scaleDataset if `TRUE`, scale the whole dataset only once instead of the train set at each replicate. Optimize processing time for predictions with large rasters.
#' @param NNmodel if TRUE, return the serialized model with the result.
#' @param DALEXexplainer if TRUE, return a explainer for the models from \code\link[DALEX]{explain}} function. It doesn't work with multisession future plans.
#' @param variableResponse if TRUE, return aggregated_profiles_explainer object from \code\link[ingredients]{partial_dependency}} and the coefficients of the adjusted linear model.
#' @param save_validateset save the validateset (independent data not used for training).
#' @param baseFilenameNN if no missing, save the NN in hdf5 format on this path with iteration appended.
#' @param filenameRasterPred if no missing, save the predictions in a RasterBrick to this file.
#' @param tempdirRaster path to a directory to save temporal raster files.
#' @param nCoresRaster number of cores used for parallelized raster cores. Use half of the available cores by default.
#' @param verbose If > 0, print state and passed to keras functions
#' @param ... extra parameters for \code\link[future.apply]{future_replicate}}  and \code\link[ingredients]{feature_importance}}.
#'
#' @return
#' @export
#' @import keras
#' @importFrom stats predict
#' @examples
pipe_keras_timeseries<- function(df, predInput=NULL, responseVars=1, caseClass=NULL, idVars=character(), weight="class",
                   timevar=NULL, responseTime="LAST", regex_time=".+", staticVars=NULL,
                   repVi=5, perm_dim=2:3, comb_dims=FALSE, crossValStrategy=c("Kfold", "bootstrap"), k=5, replicates=10, crossValRatio=c(train=0.6, test=0.2, validate=0.2),
                   hidden_shape.RNN=c(32, 32), hidden_shape.static=c(32, 32), hidden_shape.main=32, epochs=500, maskNA=NULL, batch_size="all",
                   summarizePred=TRUE, scaleDataset=FALSE, NNmodel=FALSE, DALEXexplainer=FALSE, variableResponse=FALSE, save_validateset=FALSE,
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
  if (responseTime == "LAST"){
    responseTime<- max(df[, timevar], na.rm=TRUE)
  }

  predVars<- setdiff(colnames(df), c(idVars, timevar))
  predVars.cat<- names(which(!sapply(df[, predVars, drop=FALSE], is.numeric)))
  predVars.num<- setdiff(predVars, predVars.cat)

  if (length(predVars.cat) > 0){
    df.catBin<- stats::model.matrix(stats::as.formula(paste("~ -1 +", paste(predVars.cat, collapse="+"))), data=df)
    predVars.catBin<- colnames(df.catBin)
    if (nrow(df) != nrow(df.catBin)){
      catNA<- sapply(df[, predVars.cat, drop=FALSE], function(x) any(is.na(x)) )
      catNA<- names(catNA)[catNA]
      rmRows<- unique(unlist(lapply(catNA, function(x) which(is.na(df[, x])))))
      df<- df[-rmRows, ]
      df.catBin<- stats::model.matrix(stats::as.formula(paste("~ -1 +", paste(predVars.cat, collapse="+"))), data=df)
      predVars.catBin<- colnames(df.catBin)
      warning("NAs in categorical variables not allowed. Check vars: [", paste(catNA, collapse=", "), "]. Removed rows: [", paste(rmRows, collapse=", "), "].")
    }
    df<- cbind(df[, setdiff(colnames(df), predVars.cat)], df.catBin)
    predVars<- c(predVars.num, predVars.catBin)
    staticVars.cat<- staticVars[staticVars %in% predVars.cat]
    if (all(predVars.cat %in% staticVars.cat)){
      staticVars<- c(setdiff(staticVars, staticVars.cat), predVars.catBin)
    } else {
      warning("Time varying categorical variables not supported yet.")
      staticVars<- c(setdiff(staticVars, staticVars.cat), predVars.catBin)
    }
    rm(df.catBin)
  }

  ## Select and sort predVars in predInput based on var names matching in df
  idVarsPred<- NULL
  if (!is.null(predInput)){
    if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
      ## TODO: raster predictions not implemented yet. Categorical vars ----
      selCols<- intersect(predVars, names(predInput))
      predInput<- predInput[[selCols]]
      if (!identical(selCols, names(predInput)))
        stop("Input names for predictions doesn't match input for training. Check variable names.")
    } else if (inherits(predInput, c("data.frame", "matrix"))){
      if (length(predVars.cat) > 0){
        predInput.catBin<- stats::model.matrix(stats::as.formula(paste("~ -1 +", paste(predVars.cat, collapse="+"))), data=predInput)
        predInput<- cbind(predInput[, setdiff(colnames(predInput), predVars.cat)], predInput.catBin)
        rm(predInput.catBin)
      }

      idVarsPred<- intersect(idVars, colnames(predInput))
    }
  }

  if (scaleDataset){
    ## WARNING: responseVars also scaled ----
    df.scaled<- scale(df[, predVars.num, drop=FALSE])
    df[, predVars.num]<- df.scaled
    col_means_train<- attr(df.scaled, "scaled:center")
    col_stddevs_train<- attr(df.scaled, "scaled:scale")
    rm(df.scaled)

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
        predInput[[predVars.num]]<- raster::clusterR(predInput[[predVars.num]], function(x, col_means_train, col_stddevs_train){
                              raster::calc(x, fun=function(y) scale(y, center=col_means_train, scale=col_stddevs_train))
                            }, args=list(col_means_train=col_means_train, col_stddevs_train=col_stddevs_train), filename=filenameScaled)
        if (!is.null(maskNA)){
          predInput<- raster::clusterR(predInput, function(x, maskNA){
                                raster::calc(x, fun=function(y) { y[is.na(y)]<- maskNA; y } )
                              }, args=list(maskNA=maskNA), filename=filenameScaled, overwrite=TRUE)
        }
        raster::endCluster()

      }  else if (inherits(predInput, c("data.frame", "matrix"))) {
        predInput[, names(col_means_train)]<- scale(predInput[, names(col_means_train), drop=FALSE], center=col_means_train, scale=col_stddevs_train)
      }

    }
  }

  # crossvalidation for timeseries must be done in the wide format data
  # scaling must be done on long format data
  ## NEEDS a refactor :_( or changing data formats twice or map cases from wide to long
  ## TODO: QUESTION: How to build de 3Darray? Discard responseTime VS responseTime for responseVars <- NA VS irregular shape coder/encoder layer ----
  responseVars.ts<- paste0(responseVars, "_", responseTime)
  predVars.tf<- paste0(setdiff(predVars, staticVars), "_", responseTime)
  df.wide<- longToWide.ts(d=df, timevar=timevar, idCols=c(idVars, staticVars))
  predVars.ts<- setdiff(colnames(df.wide), c(idVars, staticVars)) # WARNING: Includes responseVars.ts
  timevals<- unique(df[[timevar]])

  if (!is.null(predInput)){
    predInput.wide<- longToWide.ts(d=predInput, timevar=timevar, idCols=c(idVars, staticVars))
    predInput.wide<- predInput.wide[, c(idVars, staticVars, predVars.ts), drop=FALSE]
  }

  if (!is.null(maskNA) & scaleDataset){
    df.wide[, c(staticVars, predVars.ts)]<- apply(df.wide[, c(staticVars, predVars.ts), drop=FALSE], 2, function(x){
      x[is.na(x)]<- maskNA
      x
    })

    if (!is.null(predInput)){
      predInput.wide[, c(staticVars, predVars.ts)]<- apply(predInput.wide[, c(staticVars, predVars.ts), drop=FALSE], 2, function(x){
        x[is.na(x)]<- maskNA
        x
      })
    }
  }

  idxSetsL<- switch(crossValStrategy,
    bootstrap=bootstrap_train_test_validate(df.wide, replicates=replicates, ratio=crossValRatio, caseClass=caseClass, weight=weight),
    Kfold=kFold_train_test_validate(d=df.wide, k=k, replicates=replicates, caseClass=caseClass, weight=weight)
  )

  res<- future.apply::future_lapply(idxSetsL, function(idx.repli){
    resi<- list()
    crossValSets<- lapply(idx.repli[intersect(c("trainset", "testset"), names(idx.repli))], function(x) df.wide[x, ])

    sample_weight<- idx.repli[intersect(c("weight.train", "weight.test"), names(idx.repli))]

    # If no validation set exist, use test set to check performance
    if (length(idx.repli$validateset) > 0){
      validationSet<- df.wide[idx.repli$validateset, ]
      if (!is.null(idx.repli$weight.validate)){
        sample_weight$weight.validate<- idx.repli$weight.validate
      }
    } else {
      validationSet<- crossValSets$testset
      if (!is.null(idx.repli$weight.test)){
        sample_weight$weight.validate<- sample_weight$weight.test
      }
    }

    if (!scaleDataset){
      # NOTE: if *_data are matrix, wideToLong.ts transform timevar to character, all columns become characters.
      trainset.long<- wideToLong.ts(d=crossValSets$trainset, timevar=timevar, vars=setdiff(predVars, c(idVars, staticVars)), idCols=c(idVars, staticVars), regex_time=regex_time)
      trainset.longScaled<- scale(trainset.long[, predVars.num])
      trainset.long[, predVars.num]<- trainset.longScaled
      col_means_train<- attr(trainset.longScaled, "scaled:center")
      col_stddevs_train<- attr(trainset.longScaled, "scaled:scale")
      ## NOTE: must contain idVars and idVars should be a unique id for each case in a wide format
      trainset<- longToWide.ts(d=trainset.long, timevar=timevar, idCols=c(idVars, staticVars))

      # Scale trestset in wide format
      matchVars<- data.frame(predVars=gsub(paste0("_(", paste(timevals, collapse="|"), ")$"), "", predVars.ts),
                             predVars.ts, stringsAsFactors=FALSE)
      matchVars<- rbind(matchVars, data.frame(predVars=staticVars, predVars.ts=staticVars))
      matchVars<- merge(data.frame(predVars=names(col_means_train), col_means_train=col_means_train, col_stddevs_train=col_stddevs_train),
                        matchVars, by="predVars")
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
        predInput.wide[, matchVars$predVars.ts]<- scale(predInput.wide[, matchVars$predVars.ts],
                                                   center=matchVars[, "col_means_train"],
                                                   scale=matchVars[, "col_stddevs_train"])
        if (!is.null(maskNA)){
          predInput.wide[, selCols]<- apply(predInput.wide[, selCols], 2, function(x){
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

    validate_labels<- validationSet[, c(idVars, responseVars.ts), drop=FALSE]
    validate_data<- validationSet[, c(idVars, staticVars, predVars.ts), drop=FALSE]


    # Reshape data to 3D arrays [samples, timesteps, features] as expected by LSTM layer
    train_data.3d<- wideTo3Darray.ts(d=train_data, vars=setdiff(predVars, staticVars), idCols=idVars)
    test_data.3d<- wideTo3Darray.ts(d=test_data, vars=setdiff(predVars, staticVars), idCols=idVars)
    validate_data.3d<- wideTo3Darray.ts(d=validate_data, vars=setdiff(predVars, staticVars), idCols=idVars)

    train_labels<- structure(as.matrix(train_labels[, responseVars.ts, drop=FALSE]),
                        dimnames=list(case=do.call(paste, c(train_labels[, idVars, drop=FALSE], list(sep="_"))), var=responseVars.ts))
    test_labels<- structure(as.matrix(test_labels[, responseVars.ts, drop=FALSE]),
                        dimnames=list(case=do.call(paste, c(test_labels[, idVars, drop=FALSE], list(sep="_"))), var=responseVars.ts))
    validate_labels<- structure(as.matrix(validate_labels[, responseVars.ts, drop=FALSE]),
                        dimnames=list(case=do.call(paste, c(validate_labels[, idVars, drop=FALSE], list(sep="_"))), var=responseVars.ts))

    ## TODO: decide if responseVars.ts<- NA or remove predVars.tf ----
    if (is.null(maskNA)){
      train_data.3d<- train_data.3d[, setdiff(dimnames(train_data.3d)[[2]], responseTime), ]
      test_data.3d<- test_data.3d[, setdiff(dimnames(test_data.3d)[[2]], responseTime), ]
      validate_data.3d<- validate_data.3d[, setdiff(dimnames(validate_data.3d)[[2]], responseTime), ]
    } else {
    # VS
      train_data.3d[, as.character(responseTime), responseVars]<- maskNA
      test_data.3d[, as.character(responseTime), responseVars]<- maskNA
      validate_data.3d[, as.character(responseTime), responseVars]<- maskNA
    }

    if (length(staticVars) > 0){
      train_data.static<- structure(as.matrix(train_data[, staticVars, drop=FALSE]),
                                    dimnames=list(case=do.call(paste, c(train_data[, idVars, drop=FALSE], list(sep="_"))), var=staticVars))
      test_data.static<- structure(as.matrix(test_data[, staticVars, drop=FALSE]),
                                   dimnames=list(case=do.call(paste, c(test_data[, idVars, drop=FALSE], list(sep="_"))), var=staticVars))
      validate_data.static<- structure(as.matrix(validate_data[, staticVars, drop=FALSE]),
                                       dimnames=list(case=do.call(paste, c(validate_data[, idVars, drop=FALSE], list(sep="_"))), var=staticVars))

      train_data<- list(TS_input=train_data.3d, Static_input=train_data.static)
      test_data<- list(TS_input=test_data.3d, Static_input=test_data.static)
      validate_data<- list(TS_input=validate_data.3d, Static_input=validate_data.static)

      rm(train_data.static, test_data.static, validate_data.static)
    } else {
      train_data<- train_data.3d
      test_data<- test_data.3d
      validate_data<- validate_data.3d
    }

    ## TODO: check if reset_state is faster and equivalent to build_model. Not possible to reuse model among replicates
    ## WARNING: Don't import/export NNmodel nor python objects to code inside future for PSOCK clusters, callR.
    # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
    # modelNN<- keras::reset_states(modelNN)

    modelNN<- build_modelLTSM(input_shape.ts=dim(train_data.3d)[-1], input_shape.static=length(staticVars), output_shape=length(responseVars),
                              hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main, mask=maskNA)

    ## Check convergence on the max epochs frame
    early_stop<- keras::callback_early_stopping(monitor="val_loss", patience=30)

    ## NOTE: all data must be matrix or array or a list of if the model has multiple inputs
    modelNN<- train_keras(modelNN=modelNN, train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels,
                          epochs=epochs, batch_size=ifelse(batch_size %in% "all", nrow(train_data.3d), batch_size),
                          sample_weight=sample_weight, callbacks=early_stop, verbose=max(c(0, verbose - 2)))

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
                                         batch_size=ifelse(batch_size %in% "all", nrow(validate_data.3d), batch_size),
                                         sample_weight=sample_weight.validate, verbose=max(c(0, verbose - 2)))

    if (verbose > 1) message("Performance analyses done")

    ## Explain model
    ## TODO: fix Explain model incorrect number of dimensions. Perhaps not ready for 3d data? ----

    if (repVi > 0){
      variable_groups<- NULL
      if (length(predVars.cat) > 0){
        # Join variable importance for predictors from the same categorical variable
        v_groups.static<- lapply(predVars.cat, function(x){
          list(grep(paste0("^", x), dimnames(validate_data$Static_input)$var, value=TRUE))
        })
        v_groups.static<- stats::setNames(v_groups.static, nm=predVars.cat)
        staticNumOri<- setdiff(staticVars, predVars.catBin)
        v_groups.static<- c(stats::setNames(as.list(staticNumOri), nm=staticNumOri), v_groups.static)

        if (comb_dims){
          v_groups.ts<- expand.grid(dimnames(validate_data$TS_input)[-1], stringsAsFactors=FALSE, KEEP.OUT.ATTRS=FALSE) # All combinations for all dimensions in a dataset
          rownames(v_groups.ts)<- apply(v_groups.ts, 1, function(v) paste(v, collapse="|"))
          v_groups.ts<- split(v_groups.ts, rownames(v_groups.ts))
          v_groups.ts<- lapply(v_groups.ts, as.list)
        } else {
          v_groups.ts<- mapply(function(dimVar, dimNames) {
            v<- lapply(dimVar, function(v) stats::setNames(list(v), dimNames))
            stats::setNames(v, nm = dimVar)
          }, dimVar=dimnames(validate_data$TS_input)[-1], dimNames=names(dimnames(validate_data$TS_input))[-1], SIMPLIFY=FALSE)
          v_groups.ts<- do.call(c, v_groups.ts)
        }

        variable_groups<- list(TS_input=v_groups.ts, Static_input=v_groups.static)
      }
      resi$variableImportance<- variableImportance(model=modelNN, data=validate_data, y=validate_labels,
                                                         repVi=repVi, variable_groups=variable_groups, perm_dim=perm_dim, comb_dims=comb_dims,
                                                         batch_size=ifelse(batch_size %in% "all", nrow(validate_data.3d), batch_size), ...)
    }

    if (variableResponse | DALEXexplainer){
      ## TODO: DALEX not ready for multiinput models
      # explainer<- DALEX::explain(model=modelNN, data=validate_data, y=validate_labels, predict_function=stats::predict, label="MLP_keras", verbose=FALSE)

      ## Variable importance


      ## Variable response
      if (variableResponse){
        warning("variableResponse not implemented for 3d arrays yet. TODO: fix ingredients::partial_dependence")
        # resi$variableResponse<- variableResponse_keras(explainer)
      }

      if (DALEXexplainer){
        if (length(staticVars) > 0){
          ## TODO
          # warning("DALEX not ready for multiinput models")
        } else {
          explainer<- DALEX::explain(model=modelNN, data=validate_data, y=validate_labels, predict_function=stats::predict, label="RNN_keras", verbose=FALSE)
          resi$explainer<- explainer
        }
      }
      if (verbose > 1) message("Model analyses done")
    }


    ## Predictions
    if (!is.null(predInput)){
      if (inherits(predInput, "Raster")){
        batch_sizePred<- ifelse(batch_size %in% "all", raster::ncell(predInput), batch_size)
      } else {
        predInput.3d<- wideTo3Darray.ts(d=predInput.wide, vars=setdiff(predVars, staticVars), idCols=idVars)
        batch_sizePred<- ifelse(batch_size %in% "all", nrow(predInput.3d), batch_size)
        if (is.null(maskNA)){
          predInput.3d<- predInput.3d[, setdiff(dimnames(predInput.3d)[[2]], responseTime), ]
        } else {
          predInput.3d[, as.character(responseTime), responseVars]<- maskNA
        }
        if (length(staticVars) > 0){
          predInput.static<- structure(as.matrix(predInput.wide[, staticVars, drop=FALSE]),
                                       dimnames=list(case=do.call(paste, c(predInput.wide[, idVars, drop=FALSE], list(sep="_"))), var=staticVars))

          predInput<- list(TS_input=predInput.3d, Static_input=predInput.static)
        } else {
          predInput<- predInput.3d
        }
      }

      resi$predictions<- predict_keras(modelNN=modelNN, predInput=predInput,
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

  out<- gatherResults.pipe_result.keras(res=res, summarizePred=summarizePred, filenameRasterPred=filenameRasterPred, nCoresRaster=nCoresRaster, repNames=names(idxSetsL))
  out$params<- list(responseVars=responseVars, predVars=predVars, staticVars=staticVars, predVars.cat=predVars.cat,
                    caseClass=caseClass, idVars=idVars, weight=weight,
                    timevar=timevar, responseTime=responseTime, regex_time=regex_time, perm_dim=perm_dim, comb_dims=comb_dims,
                    repVi=repVi, crossValStrategy=crossValStrategy, k=k, replicates=replicates, crossValRatio=crossValRatio,
                    shapeNN=list(hidden_shape.RNN=hidden_shape.RNN, hidden_shape.static=hidden_shape.static, hidden_shape.main=hidden_shape.main),
                    epochs=epochs, maskNA=maskNA, batch_size=batch_size,
                    summarizePred=summarizePred, scaleDataset=scaleDataset, NNmodel=NNmodel, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse,
                    save_validateset=save_validateset, baseFilenameNN=baseFilenameNN, filenameRasterPred=filenameRasterPred)
  if (crossValStrategy != "Kfold") out$params$k<- NULL

  if (!is.null(predInput) & inherits(predInput, c("data.frame", "matrix")) & length(idVarsPred) > 0){
    ## Add idVars if exists
    predInputIdVars<- predInput[, idVarsPred, drop=FALSE]
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

  if (save_validateset){
    out$validateset<- lapply(idxSetsL, function(x) df[x$validateset, ])
  }

  class(out)<- c("pipe_result.keras", "pipe_result")

  return(out)
}
