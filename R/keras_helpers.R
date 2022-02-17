#' Neural network model with keras
#'
#' @param df a data.frame with response variable in the first column
#' @param predInput a Raster or a data.frame with columns 1 and 2 corresponding to longitude and latitude + variables for the model
#' @param responseVars response variables. Column names or indexes on df.
#' @param caseClass class of the samples used to weight cases. Column names or indexes on df, or a vector with the class for each rows in df.
#' @param idVars id column names or indexes on df and/or predInput. Deprecated default for compatibility c("x", "y")
#' @param weight Optional array of the same length as \code{nrow(df)}, containing weights to apply to the model's loss for each sample.
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
process_keras<- function(df, predInput=NULL, responseVars=1, caseClass=NULL, idVars=character(), weight="class",
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
        predInput<- scale(predInput[, , drop=FALSE], center=col_means_train, scale=col_stddevs_train)
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
    # crossValSets<- NNTools:::splitdf(df, ratio=crossValRatio, sample_weight=sample_weight)
    crossValSets<- lapply(idx.repli[intersect(c("trainset", "testset"), names(idx.repli))], function(x) df[x, ])

    train_labels<- as.matrix(crossValSets$trainset[, responseVars, drop=FALSE])
    train_data<- as.matrix(crossValSets$trainset[, predVars, drop=FALSE])

    test_labels<- as.matrix(crossValSets$testset[, responseVars, drop=FALSE])
    test_data<- as.matrix(crossValSets$testset[, predVars, drop=FALSE])

    sample_weight<- idx.repli[intersect(c("weight.train", "weight.test"), names(idx.repli))]
    # if (length(sample_weight) == 0) sample_weight<- NULL

    # If no validation set exist, use test set to check performance
    if (length(idxSetsL$validateset) > 0){
      validate_labels<- as.matrix(df[idxSetsL$validateset, responseVars, drop=FALSE])
      validate_data<- as.matrix(df[idxSetsL$validateset, predVars, drop=FALSE])
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
      train_data<- scale(train_data)

      col_means_train<- attr(train_data, "scaled:center")
      col_stddevs_train<- attr(train_data, "scaled:scale")

      test_data<- scale(test_data, center=col_means_train, scale=col_stddevs_train)
      validate_data<- scale(validate_data, center=col_means_train, scale=col_stddevs_train)

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
    }

    resi$scaleVals<- data.frame(mean=col_means_train, sd=col_stddevs_train)

    ## TODO: check if reset_state is faster and equivalent to build_model. Not possible to reuse model among replicates
    ## WARNING: Don't import/export NNmodel nor python objects to code inside future for PSOCK clusters, callR.
    # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
    modelNN<- NNTools:::build_modelDNN(input_shape=length(predVars), output_shape=length(responseVars), hidden_shape=hidden_shape, mask=maskNA)
    # modelNN<- keras::reset_states(modelNN)

    ## Check convergence on the max epochs frame
    early_stop<- keras::callback_early_stopping(monitor="val_loss", patience=30)

    modelNN<- NNTools:::train_keras(modelNN=modelNN, train_data=train_data, train_labels=train_labels,
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
    resi$performance<- NNTools:::performance_keras(modelNN=modelNN, test_data=validate_data, test_labels=validate_labels,
                             batch_size=ifelse(batch_size %in% "all", nrow(test_data), batch_size),
                             sample_weight=sample_weight.validate, verbose=verbose)

    if (verbose > 1) message("Performance analyses done")

    ## Explain model
    if (repVi > 0 | variableResponse | DALEXexplainer){
      explainer<- DALEX::explain(model=modelNN, data=validate_data, y=validate_labels, predict_function=stats::predict, label="MLP_keras", verbose=FALSE)

      ## Variable importance
      if (repVi > 0){
        resi$variableImportance<- NNTools:::variableImportance_keras(explainer=explainer, repVi=repVi)
      }
      ## Variable response
      if (variableResponse){
        resi$variableResponse<- NNTools:::variableResponse_keras(explainer)
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

      resi$predictions<- NNTools:::predict_keras(modelNN=modelNN, predInput=predInput, maskNA=maskNA,
                               scaleInput=!scaleDataset, col_means_train=col_means_train, col_stddevs_train=col_stddevs_train,
                               batch_size=batch_sizePred, tempdirRaster=tempdirRaster, nCoresRaster=nCoresRaster)
      if (inherits(resi$predictions, "matrix")){
        colnames(resi$predictions)<- responseVars
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


  ## Gather results
  names(res)<- paste0("rep", formatC(1:length(res), format="d", flag="0", width=nchar(length(res))))

  out<- list(performance=do.call(rbind, lapply(res, function(x) x$performance)),
             scale=lapply(res, function(x) x$scaleVals))

  if (!is.null(res[[1]]$variableImportance)){
    vi<- lapply(res, function(x){
            tmp<- x$variableImportance
            tmp[sort(rownames(tmp)), ]
          })
    nPerm<- ncol(vi[[1]])
    vi<- do.call(cbind, vi)

    out$vi<- vi[order(rowSums(vi)), , drop=FALSE] ## Order by average vi
    colnames(out$vi)<- paste0(rep(paste0("rep", formatC(1:length(res), format="d", flag="0", width=nchar(length(res)))), each=nPerm),
                              "_", colnames(out$vi))
  }

  if (!is.null(res[[1]]$variableResponse)){
    out$variableResponse<- lapply(res, function(x){
      x$variableResponse$variableResponse
    })

    variableCoef<- lapply(res, function(x){
      x$variableResponse$var_coef
    })

    out$variableCoef<- lapply(rownames(variableCoef[[1]]), function(x){
      varCoef<- lapply(variableCoef, function(y){
        stats::na.omit(y[x, ])
      })

      degrees<- sapply(varCoef, function(y) y["degree"])
      maxDegree<- max(degrees)

      if (length(unique(degrees)) > 1){
        # Add NA if varCoef elements have degree < maxDegree (different length)
        sel<- degrees < maxDegree
        varCoef[sel]<- lapply(varCoef[sel], function(x){
                  c(x[1:(1 + x["degree"])], rep(NA_real_, maxDegree - x["degree"]), x[c("adj.r.squared", "r.squared", "degree")])
                })
      }

      structure(do.call(rbind, varCoef),
                dimnames=list(names(varCoef), ## TODO: check translation response var from ingredients::partial_dependency()$`_label_`
                              c("intercept", paste0("b", 1:(maxDegree)), "adj.r.squared", "r.squared", "degree")))
    })
    names(out$variableCoef)<- rownames(variableCoef[[1]])
  }

  ## Predictions
  if (!is.null(res[[1]]$predictions)){
    out$predictions<- lapply(res, function(x) x$predictions)

    if (inherits(res[[1]]$predictions, "Raster")){
      resVarNames<- names(out$predictions[[1]])
      out$predictions<- raster::stack(out$predictions)
      lnames<- paste0(rep(resVarNames, times=length(res)),
                      rep(paste0("_rep", formatC(1:length(res), format="d", flag="0", width=nchar(length(res)))),
                          each=length(resVarNames)))
      names(out$predictions)<- lnames

      if (!is.null(filenameRasterPred)){
        if (summarizePred){
          out$predictions<- summarize_pred(pred=out$predictions, filename=filenameRasterPred, nCoresRaster=nCoresRaster)
        } else {
          out$predictions<- raster::brick(out$predictions, filename=filenameRasterPred)
        }
      } else {
        if (summarizePred){
          out$predictions<- summarize_pred(pred=out$predictions, nCoresRaster=nCoresRaster)
        } else {
          out$predictions<- raster::brick(out$predictions)
        }

        if (!raster::inMemory(out$predictions)){
          warning("The rasters with the predictions doesn't fit in memory and the values are saved in a temporal file. ",
                  "Please, provide the filenameRasterPred parameter to save the raster in a non temporal file. ",
                  "If you want to save the predictions of the current run use writeRaster on result$predicts before to close the session.")
        }
      }

      tmpFiles<- sapply(res, function(x){
          raster::filename(x$predictions)
        })
      tmpFiles<- tmpFiles[tmpFiles != ""]

      file.remove(tmpFiles, gsub("\\.grd", ".gri", tmpFiles))

    } else { ## non Raster predInput
      resVarNames<- colnames(out$predictions[[1]])
      out$predictions<- lapply(resVarNames, function(x){
                          do.call(cbind, lapply(out$predictions, function(y) y[, x, drop=FALSE]))
                        })
      names(out$predictions)<- resVarNames

      if (summarizePred){
        out$predictions<- lapply(out$predictions, function(x){
                            rownames(x)<- rownames(predInput)
                            summarize_pred.default(x)
                          })
      }else{
        out$predictions<- lapply(out$predictions, function(x){
                              rownames(x)<- rownames(predInput)
                              colnames(x)<- paste0("rep", formatC(1:length(res), format="d", flag="0", width=nchar(length(res))))
                              x
                           })
      }

    }

  }


  if (!is.null(res[[1]]$model)){
    out$model<- lapply(res, function(x){
      x$model # unserialize_model() to use the saved model
    })
  }

  if (!is.null(res[[1]]$explainer)){
    out$DALEXexplainer<- lapply(res, function(x){
      x$explainer
    })
  }

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


train_keras<- function(modelNN, train_data, train_labels, test_data, test_labels, epochs, batch_size, sample_weight=NULL, callbacks=NULL, verbose=0){
  if (is.null(sample_weight)){
    validation_data<- list(test_data, test_labels)
  } else {
    validation_data<- list(test_data, test_labels, sample_weight$weight.test)
    sample_weight<- sample_weight$weight.train
  }

  history<- keras::fit(object=modelNN,
                x=train_data,
                y=train_labels,
                batch_size=ifelse(batch_size %in% "all", nrow(train_data), batch_size),
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                sample_weight=sample_weight
  )

  return(modelNN)
}


# @importFrom caret postResample sample_weight
performance_keras<- function(modelNN, test_data, test_labels, batch_size, sample_weight=NULL, verbose=0){
  perf<- keras::evaluate(object=modelNN, x=test_data, y=test_labels, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight)
  perfCaret<- caret::postResample(pred=stats::predict(modelNN, test_data), obs=test_labels) # No weighting

  out<- data.frame(as.list(perf), as.list(perfCaret))
  return(out)
}


# @importFrom DALEX explain
# @importFrom ingredients feature_importance
variableImportance_keras<- function(explainer, repVi=5){
  if (repVi > 0){
    vi<- ingredients::feature_importance(explainer, B=repVi)
    vi<- stats::reshape(as.data.frame(vi)[, c("variable", "dropout_loss", "permutation")], timevar="permutation", idvar="variable", direction="wide")
    vi<- structure(as.matrix(vi[, -1]),
                   dimnames=list(as.character(vi$variable),
                     paste0("perm", formatC(0:(ncol(vi) -2), format="d", flag="0", width=nchar(repVi)))))
  } else {
    vi<- NA
  }

  return(vi)
}


variableResponse_keras<- function(explainer, variables=NULL, maxPoly=5){
  if (is.null(variables)){
    variables<- colnames(explainer$data)
  }

  varResp<- ingredients::partial_dependency(x=explainer, variables=variables, variable_type="numerical")

  ## TODO: check that thesaurus linking response variable names from ingredients::partial_dependency & from original data is correct
  thesaurusResp<- data.frame(respOri=colnames(explainer$y), respIngredients=unique(varResp$`_label_`), stringsAsFactors=FALSE)
  var_coefsL<- list()

  var_coefsL<- by(varResp, paste(varResp$`_label_`, varResp$`_vname_`), function(x){
                  for (deg in 1:maxPoly){
                    mvar<- stats::lm(`_yhat_` ~ poly(`_x_`, deg, raw=TRUE), data=x)
                    smvar<- summary(mvar)

                    if (smvar$adj.r.squared > 0.9) {
                      break
                    }
                  }

                  # Translate response name to the original
                  form<- paste(merge(x[1, "_label_"], thesaurusResp, by.x="x", by.y="respIngredients")$respOri, "~", x[1, "_vname_"])
                  list(formula=form, coefficients=stats::coef(mvar), degree=deg,
                       fit=c(adj.r.squared=smvar$adj.r.squared, r.squared=smvar$r.squared))
  }, simplify=FALSE)

  # Build a matrix with NAs in missing colums
  maxNcoef<- max(sapply(var_coefsL, function(x) length(x$coefficients)))
  var_coefs<- structure(t(sapply(var_coefsL, function(x){
                    c(x$coefficients, rep(NA_real_, maxNcoef - length(x$coefficients)),
                      x$fit, x$degree)
                })), dimnames=list(sapply(var_coefsL, function(x) x$formula),
                                 c("intercept", paste0("b", 1:(maxNcoef - 1)), "adj.r.squared", "r.squared", "degree"))
              )

  return(list(var_coefs=var_coefs, variableResponse=varResp))
}


## FUNCTIONS: Build and Train Neural Networks ----
# 2 hidden layers
build_modelDNN<- function(input_shape, output_shape=1, hidden_shape=128, mask=NULL){
  if (is.null(mask)){
    model<- keras_model_sequential() %>%
      layer_dense(units=hidden_shape, activation="relu", input_shape=input_shape) %>%
      layer_dense(units=hidden_shape, activation="relu") %>%
      layer_dense(units=output_shape)
  } else {
    model<- keras_model_sequential() %>%
      layer_masking(mask_value=mask, input_shape=input_shape) %>%
      layer_dense(units=hidden_shape, activation="relu") %>%
      layer_dense(units=hidden_shape, activation="relu") %>%
      layer_dense(units=output_shape)
  }
  compile(model,
          loss="mse",
          optimizer=optimizer_rmsprop(),
          metrics=list("mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error")
  )

  model
}


# https://github.com/rspatial/raster/blob/b1c9d91b1b43b17ea757889dc93f97bd70dc1d2e/R/predict.R
# ?raster::`predict,Raster-method`
# ?predict.keras.engine.training.Model
predict.Raster_keras<- function(object, model, filename="", fun=predict, ...) {
  nLayersOut<- model$output_shape[[2]]
  out<- raster::brick(object, values=FALSE, nl=nLayersOut)
  big<- !raster::canProcessInMemory(out, raster::nlayers(object) + nLayersOut)
  filename<- raster::trim(filename)

  if (big & filename == "") {
    filename<- raster::rasterTmpFile()
  }

  if (filename != "") {
    out<- raster::writeStart(out, filename, ...)
    todisk<- TRUE
  } else {
    # ncol=nrow(), nrow=ncol() from https://rspatial.org/raster/pkg/appendix1.html#a-complete-function
    vv<- array(NA_real_, dim=c(ncol(out), nrow(out), raster::nlayers(out)))
    todisk<- FALSE
  }

  bs<- raster::blockSize(object)
  pb<- raster::pbCreate(bs$n)

  if (todisk) {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- fun(object=model, v, ...)

      out<- raster::writeValues(out, v, bs$row[i])
      raster::pbStep(pb, i)
    }

    out<- raster::writeStop(out)
  } else {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- fun(object=model, v, ...)

      cols<- bs$row[i]:(bs$row[i] + bs$nrows[i] - 1)
      vv[, cols, ]<- array(v, dim=c(ncol(object), length(cols), nLayersOut))

      raster::pbStep(pb, i)
    }

    out<- raster::setValues(out, as.vector(vv))
  }

  raster::pbClose(pb)

  return(out)
}


#' Title
#'
#' @param modelNN
#' @param predInput data.frame or raster with colnames or layer names matching the expected input for modelNN
#' @param maskNA
#' @param scaleInput
#' @param col_means_train
#' @param col_stddevs_train
#' @param batch_size
#' @param filename
#' @param tempdirRaster
#'
#' @return
#'
#' @examples
predict_keras<- function(modelNN, predInput, maskNA=NULL, scaleInput=FALSE, col_means_train, col_stddevs_train, batch_size, filename="", tempdirRaster=NULL, nCoresRaster=2){
  if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
    if (!is.null(tempdirRaster)){
      filenameScaled<- tempfile(tmpdir=tempdirRaster, fileext=".grd")
      if (filename == ""){
        filename<- tempfile(tmpdir=tempdirRaster, fileext=".grd")
      }
    } else {
      filenameScaled<- raster::rasterTmpFile()
    }

    if (scaleInput){
      # predInputScaled<- raster::scale(predInput, center=col_means_train, scale=col_stddevs_train)
      raster::beginCluster(n=nCoresRaster)
      predInputScaled<- raster::clusterR(predInput, function(x, col_means_train, col_stddevs_train){
                                    raster::calc(x, fun=function(y) scale(y, center=col_means_train, scale=col_stddevs_train))
                                  }, args=list(col_means_train=col_means_train, col_stddevs_train=col_stddevs_train), filename=filenameScaled)
      if (!is.null(maskNA)){
        predInputScaled<- raster::clusterR(predInputScaled, function(x, maskNA){
                                      raster::calc(x, fun=function(y) { y[is.na(y)]<- maskNA; y } )
                                    }, args=list(maskNA=maskNA), filename=filenameScaled, overwrite=TRUE) # gsub(".grd$", "_mask.grd", filenameScaled)
      }
      raster::endCluster()
      names(predInputScaled)<- names(predInput)
      # ?keras::predict.keras.engine.training.Model
      # ?raster::`predict,Raster-method`
    } else {
      predInputScaled<- predInput
    }

    predicts<- predict.Raster_keras(object=predInputScaled, model=modelNN, filename=filename,
                                    batch_size=batch_size) # TODO, verbose=verbose)

    if (scaleInput){
      file.remove(filenameScaled, gsub("\\.grd$", ".gri", filenameScaled))
      rm(predInputScaled)
    }

    # predicti<- raster::predict(object=predInputScaled, model=modelNN,
    #                    fun=keras:::predict.keras.engine.training.Model,
    #                    filename=filename,
    #                    batch_size=ifelse(batch_size %in% "all", raster::ncell(predInputScaled), batch_size),
    #                    verbose=verbose)
    # predicts[[i]]<- predicti

  } else if (inherits(predInput, c("data.frame", "matrix"))) {
    if (scaleInput){
      predInputScaled<- scale(predInput[, , drop=FALSE],
                            center=col_means_train, scale=col_stddevs_train)
      if (!is.null(maskNA)){
        predInputScaled<- apply(predInputScaled, 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
      }
    } else {
      predInputScaled<- predInput
    }

    predicts<- stats::predict(modelNN, predInputScaled,
                              batch_size=batch_size) # TODO, verbose=verbose)
  }

  return(predicts)
}
