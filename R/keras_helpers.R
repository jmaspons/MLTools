#' Title
#'
#' @param df a data.frame whith response variable in the first column
#' @param predInput a Raster or a data.frame with columns 1 and 2 corresponding to longitude and latitude + variables for the model
#' @param responseVars response variables. Column names or indexes on df
#' @param idVars id column names or indexes on df and/or predInput. Deprecated default for compatibility c("x", "y")
#' @param replicates number of replicates
#' @param repVi replicates of the permutations to calculate the importance of the variables. 0 to avoid report variable importance
#' @param crossValRatio Proportion of the dataset used to train the model. Default to 0.8
#' @param hidden_shape number of neurons in the hidden layers of the nerual network model.
#' @param epochs parameter for \code\link[keras]{fit}}.
#' @param batch_size for fit and predict functions. The bigger the better if it fits your available memory. Integer or "all".
#' @param summarizePred if \code{TRUE}, return the mean, sd and se of the predictors. if \code{FALSE}, return the predictions for each replicate.
#' @param scaleDataset if \code{TRUE}, scale the whole dataset only once instead of the train set at each replicate. Optimize processing time for predictions with large rasters.
#' @param NNmodel if TRUE, return the serialized model with the result.
#' @param DALEXexplainer if TRUE, return a explainer for the models from \code\link[DALEX]{explain}} function. It doesn't work with multisession future plans.
#' @param variableResponse if TRUE, return aggregated_profiles_explainer object from \code\link[ingredients]{partial_dependency}} and the coefficients of the adjusted lineal model.
#' @param baseFilenameNN if no missing, save the NN in hdf5 format on this path with iteration appended.
#' @param filenameRasterPred if no missing, save the predictions in a RasterBrick format to this file.
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
process_keras<- function(df, predInput, responseVars=1, idVars=character(),
                   replicates=10, repVi=5, crossValRatio=0.8, hidden_shape=50, epochs=500, batch_size="all",
                   summarizePred=TRUE, scaleDataset=FALSE, NNmodel=FALSE, DALEXexplainer=FALSE, variableResponse=TRUE,
                   baseFilenameNN, filenameRasterPred, tempdirRaster, nCoresRaster=parallel::detectCores() %/% 2, verbose=0, ...){
  if (is.character(responseVars)){
    responseVars<- which(colnames(df) %in% responseVars)
  }
  if (is.character(idVars)){
    idVars<- which(colnames(df) %in% idVars)
  }
  predVars<- setdiff(1:ncol(df), c(responseVars, idVars))

  ## Avoid missing checks in futures
  if (missing(predInput)) predInput<- NULL
  if (missing(baseFilenameNN)) baseFilenameNN<- NULL
  if (missing(filenameRasterPred)) filenameRasterPred<- NULL
  if (missing(tempdirRaster)) tempdirRaster<- NULL

  ## Select and sort predVars in predInput based on var names matching in df
  if (!is.null(predInput)){
    if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
      selCols<- intersect(colnames(df)[predVars], names(predInput))
      predInput<- predInput[[selCols]]
      if (!identical(selCols, names(predInput)))
        stop("Input names for predictions doesn't match input for training. Check variable names.")
    }  else if (inherits(predInput, c("data.frame", "matrix"))) {
      selCols<- intersect(colnames(df)[predVars], colnames(predInput))
      idVarsPred<- intersect(colnames(df)[idVars], colnames(predInput))

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
        raster::endCluster()

      }  else if (inherits(predInput, c("data.frame", "matrix"))) {
        predInput<- scale(predInput[, , drop=FALSE],
                          center=col_means_train, scale=col_stddevs_train)
      }

    }
  }

  res<- future.apply::future_replicate(replicates, {
    resi<- list()
    crossValSets<- NNTools:::splitdf(df, ratio=crossValRatio)

    train_labels<- as.matrix(crossValSets$trainset[, responseVars, drop=FALSE])
    train_data<- as.matrix(crossValSets$trainset[, predVars, drop=FALSE])

    test_labels<- as.matrix(crossValSets$testset[, responseVars, drop=FALSE])
    test_data<- as.matrix(crossValSets$testset[, predVars, drop=FALSE])

    if (!scaleDataset){
      train_data<- scale(train_data)

      col_means_train<- attr(train_data, "scaled:center")
      col_stddevs_train<- attr(train_data, "scaled:scale")

      test_data<- scale(test_data, center=col_means_train, scale=col_stddevs_train)
    }

    resi$scaleVals<- data.frame(mean=col_means_train, sd=col_stddevs_train)

    ## TODO: check if reset_state is faster and equivalent to build_model. Not possible to reuse model among replicates
    ## WARNING: Don't import/export NNmodel nor python objects to code inside future for PSOCK clusters, callR.
    # https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html
    modelNN<- NNTools:::build_modelDNN(input_shape=length(predVars), output_shape=length(responseVars), hidden_shape=hidden_shape)
    # modelNN<- keras::reset_states(modelNN)

    ## Check convergence on the max epochs frame
    early_stop<- keras::callback_early_stopping(monitor="val_loss", patience=30)

    modelNN<- NNTools:::train_keras(modelNN=modelNN, train_data=train_data, train_labels=train_labels,
                           test_data=test_data, test_labels=test_labels, epochs=epochs,
                           batch_size=batch_size, callbacks=early_stop, verbose=verbose)

    if (NNmodel){
      resi$model<- keras::serialize_model(modelNN)
    }

    ## Model performance
    resi$performance<- NNTools:::performance_keras(modelNN=modelNN, test_data=test_data, test_labels=test_labels,
                             batch_size=ifelse(batch_size %in% "all", nrow(test_data), batch_size), verbose=verbose)

    ## Explain model
    if (repVi > 0 | variableResponse | DALEXexplainer){
      explainer<- DALEX::explain(model=modelNN, data=train_data, y=train_labels, predict_function=stats::predict, label="MLP_keras", verbose=FALSE)

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
    }

    ## Predictions
    if (!is.null(predInput)){
      if (inherits(predInput, "Raster")){
        batch_sizePred<- ifelse(batch_size %in% "all", raster::ncell(predInput), batch_size)
      } else {
        batch_sizePred<- ifelse(batch_size %in% "all", nrow(predInput), batch_size)
      }

      resi$predictions<- NNTools:::predict_keras(modelNN=modelNN, predInput=predInput, scaleInput=!scaleDataset,
                               col_means_train=col_means_train, col_stddevs_train=col_stddevs_train,
                               batch_size=batch_sizePred, tempdirRaster=tempdirRaster)
    }

    return(resi)
  }, simplify=FALSE, ...)

  if (verbose > 0) message("Iterations finished. Gathering results...")


  ## Gather results
  names(res)<- paste0("rep", formatC(1:replicates, format="d", flag="0", width=nchar(replicates)))

  out<- list(performance=do.call(rbind, lapply(res, function(x) x$performance)))

  if (scaleDataset){
    out<- c(out, list(scale=list(dataset=data.frame(mean=col_means_train, sd=col_stddevs_train))))
  } else {
    out<- c(out, list(scale=lapply(res, function(x) x$scaleVals)))
  }


  if (repVi > 0){
    vi<- lapply(res, function(x){
            tmp<- x$variableImportance
            tmp[sort(rownames(tmp)), ]
          })
    nPerm<- ncol(vi[[1]])
    vi<- do.call(cbind, vi)

    out$vi<- vi[order(rowSums(vi)), , drop=FALSE] ## Order by average vi
    colnames(out$vi)<- paste0(rep(paste0("rep", formatC(1:replicates, format="d", flag="0", width=nchar(replicates))), each=nPerm),
                              "_", colnames(out$vi))
  }

  if (variableResponse){
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
  if (!is.null(predInput)){
    out$predictions<- lapply(res, function(x) x$predictions)

    if (inherits(predInput, "Raster")){

      if (scaleDataset) file.remove(filenameScaled, gsub("\\.grd$", ".gri", filenameScaled))

      out$predictions<- raster::stack(out$predictions)

      lnames<- paste0(rep(colnames(df)[responseVars], times=replicates),
                      rep(paste0("_rep", formatC(1:replicates, format="d", flag="0", width=nchar(replicates))),
                          each=length(responseVars)))
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

      out$predictions<- lapply(seq_along(responseVars), function(x){
                          do.call(cbind, lapply(out$predictions, function(y) y[, x, drop=FALSE]))
                        })
      names(out$predictions)<- colnames(df)[responseVars]

      if (summarizePred){
        out$predictions<- lapply(out$predictions, function(x){
                            rownames(x)<- rownames(predInput)
                            summarize_pred(x)
                          })
      }else{
        out$predictions<- lapply(out$predictions, function(x){
                              rownames(x)<- rownames(predInput)
                              colnames(x)<- paste0("rep", formatC(1:replicates, format="d", flag="0", width=nchar(replicates)))
                              x
                           })
      }

      if (length(idVarsPred) > 0){
        ## Add idVars if exists
        out$predictions<- lapply(out$predictions, function(x){
                            cbind(predInputIdVars, x)
                          })
      }

    }

  }


  if (NNmodel){
    out$model<- lapply(res, function(x){
      x$model # unserialize_model() to use the saved model
    })

    if (!is.null(baseFilenameNN)){
      tmp<- lapply(seq_along(out$model), function(i){
        save_model_hdf5(keras::unserialize_model(out$model[[i]]), filepath=paste0(baseFilenameNN, "_", formatC(i, format="d", flag="0", width=nchar(replicates)), ".hdf5"),
                        overwrite=TRUE, include_optimizer=TRUE)
      })
    }
  }

  if (DALEXexplainer){
    out$DALEXexplainer<- lapply(res, function(x){
      x$explainer
    })
  }

  class(out)<- "process_NN"

  return(out)
}


train_keras<- function(modelNN, train_data, train_labels, test_data, test_labels, epochs, batch_size, callbacks=NULL, verbose=0){
  history<- keras::fit(object=modelNN,
                x=train_data,
                y=train_labels,
                epochs=epochs,
                validation_data=list(test_data, test_labels),
                verbose=verbose,
                batch_size=ifelse(batch_size %in% "all", nrow(train_data), batch_size),
                callbacks=callbacks
  )

  return(modelNN)
}


# @importFrom caret postResample
performance_keras<- function(modelNN, test_data, test_labels, batch_size, verbose=0){
  perf<- keras::evaluate(modelNN, test_data, test_labels, verbose=verbose,
                  batch_size=batch_size)
  perfCaret<- caret::postResample(pred=stats::predict(modelNN, test_data), obs=test_labels)

  out<- data.frame(data.frame(perf), as.list(perfCaret))
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
build_modelDNN<- function(input_shape, output_shape=1, hidden_shape=128){
  model<- keras_model_sequential() %>%
    layer_dense(units=hidden_shape, activation="relu", input_shape=input_shape) %>%
    layer_dense(units=hidden_shape, activation="relu") %>%
    layer_dense(units=output_shape)

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
predict_keras<- function(modelNN, predInput, scaleInput=FALSE, col_means_train, col_stddevs_train, batch_size, filename="", tempdirRaster=NULL){
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
      predInputScaled<- raster::calc(predInput, filename=filenameScaled,
                                  fun=function(x) scale(x, center=col_means_train, scale=col_stddevs_train))
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
    } else {
      predInputScaled<- predInput
    }

    predicts<- stats::predict(modelNN, predInputScaled,
                              batch_size=batch_size) # TODO, verbose=verbose)
  }

  return(predicts)
}
