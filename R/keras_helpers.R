
train_keras<- function(modelNN, train_data, train_labels, test_data, test_labels, epochs, batch_size=NULL, sample_weight=NULL, callbacks=NULL, verbose=0){
  if (is.null(sample_weight) | length(sample_weight) == 0){
    validation_data<- list(test_data, test_labels)
    sample_weight<- NULL
  } else {
    validation_data<- list(test_data, test_labels, sample_weight$weight.test)
    sample_weight<- sample_weight$weight.train
  }

  history<- keras::fit(object=modelNN,
                x=train_data,
                y=train_labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                sample_weight=sample_weight
  )

  return(modelNN)
}


# @importFrom caret postResample sample_weight
performance_keras<- function(modelNN, test_data, test_labels, batch_size=NULL, sample_weight=NULL, verbose=0){
  perf<- keras::evaluate(object=modelNN, x=test_data, y=test_labels, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight)
  perfCaret<- caret::postResample(pred=stats::predict(modelNN, test_data, batch_size=batch_size), obs=test_labels) # No weighting

  out<- data.frame(as.list(perf), as.list(perfCaret))
  return(out)
}


#' Variable importance by permutations on predictors
#'
#' @param model
#' @param data
#' @param y
#' @param repVi
#' @param variable_groups
#' @param perm_dim
#' @param comb_dims
#' @param batch_size
#' @param ...
#'
#' @return
#' @export
#'
#' @examples
variableImportance<- function(model, data, y, repVi=5, variable_groups=NULL, perm_dim=NULL, comb_dims=FALSE, batch_size=NULL, ...){
  if (repVi > 0){
    vi<- feature_importance(x=model, data=data, y=y, B=repVi, variable_groups=variable_groups,
                                         perm_dim=perm_dim, comb_dims=comb_dims, predict_function=stats::predict, batch_size=batch_size, ...)
    # vi[, "permutation" == 0] -> average; vi[, "permutation" > 0] -> replicates
    vi<- stats::reshape(as.data.frame(vi)[vi[, "permutation"] > 0, c("variable", "dropout_loss", "permutation")], timevar="permutation", idvar="variable", direction="wide")
    vi<- structure(as.matrix(vi[, -1]),
                   dimnames=list(as.character(vi$variable),
                                 paste0("perm", formatC(1:(ncol(vi) -1), format="d", flag="0", width=nchar(repVi)))))
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

  var_coefsL<- by(varResp, paste(varResp$`_label_`, varResp$`_vname_`), function(x){
                  # Translate response name to the original
                  form<- paste(merge(x[1, "_label_"], thesaurusResp, by.x="x", by.y="respIngredients")$respOri, "~", x[1, "_vname_"])
                  if (nrow(x) > 1){
                    for (deg in 1:maxPoly){
                      mvar<- stats::lm(`_yhat_` ~ poly(`_x_`, deg, raw=TRUE), data=x)
                      smvar<- summary(mvar)

                      if (smvar$adj.r.squared > 0.9 | !is.finite(smvar$adj.r.squared)) {
                        break
                      }
                    }
                    out<- list(formula=form, coefficients=stats::coef(mvar), degree=deg,
                               fit=c(adj.r.squared=smvar$adj.r.squared, r.squared=smvar$r.squared))
                  } else {
                    out<- list(formula=form, coefficients=c(`(Intercept)`=NA), degree=0,
                               fit=c(adj.r.squared=NA, r.squared=NA))
                  }

                  return(out)
  }, simplify=FALSE)

  # Build a matrix with NAs in missing colums
  maxNcoef<- max(sapply(var_coefsL, function(x) length(x$coefficients)))
  if (maxNcoef > 1){
    coefs<- paste0("b", 1:(maxNcoef - 1))
  } else {
    coefs<- character()
  }
  colNames<- c("intercept", coefs, "adj.r.squared", "r.squared", "degree")
  var_coefs<- structure(t(sapply(var_coefsL, function(x){
                    c(x$coefficients, rep(NA_real_, maxNcoef - length(x$coefficients)),
                      x$fit, x$degree)
                })), dimnames=list(sapply(var_coefsL, function(x) x$formula), colNames)
              )

  return(list(var_coefs=var_coefs, variableResponse=varResp))
}


gatherResults.pipe_result.keras<- function(res, summarizePred, filenameRasterPred, nCoresRaster){
  names(res)<- paste0("rep", formatC(1:length(res), format="d", flag="0", width=nchar(length(res))))

  out<- list(performance=do.call(rbind, lapply(res, function(x) x$performance)))
  sc<- lapply(res, function(x) {
    s<- x$scaleVals
  })
  sc<- sc[!sapply(sc, is.null)]
  if (length(sc) == 1 & all(names(sc[[1]]) == "dataset")){
    sc<- sc[[1]]
  }
  out$scale<- sc

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
        stats::na.omit(y[x, , drop=TRUE])
      })

      degrees<- sapply(varCoef, function(y) y["degree"])
      maxDegree<- max(degrees)

      if (maxDegree > 0){
        colNames<- c("intercept", paste0("b", 1:(maxDegree)), "adj.r.squared", "r.squared", "degree")
        if (length(unique(degrees)) > 1){
          # Add NA if varCoef elements have degree < maxDegree (different length)
          sel<- degrees < maxDegree
          varCoef[sel]<- lapply(varCoef[sel], function(x){
                    structure(c(x[1:(1 + x["degree"])], rep(NA_real_, maxDegree - x["degree"]), x[c("adj.r.squared", "r.squared", "degree")]),
                              names=colNames)
                  })
        }
      } else {
        colNames<- c("intercept", "adj.r.squared", "r.squared", "degree")
        varCoef<- lapply(varCoef, function(x){
          c(intercept=NA, adj.r.squared=NA, r.squared=NA, x["degree"])
        })
      }

      structure(do.call(rbind, c(varCoef, list(deparse.level=0))),
                dimnames=list(names(varCoef), colNames))## TODO: check translation response var from ingredients::partial_dependency()$`_label_`

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
                            summarize_pred.default(x)
                          })
      }else{
        out$predictions<- lapply(out$predictions, function(x){
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

  class(out)<- "pipe_result.keras"

  return(out)
}


#' @export
print.pipe_result.keras<- function(x, ...){
  cat("Keras pipe result with", nrow(x$performance), "replicates.\n")
  cat("\nPerformance:\n")
  print(x$performance, ...)
  if (!is.null(x$vi)){
    cat("\nVariable Importance:\n")
    print(x$vi, ...)
  }
  if (!is.null(x$variableCoef)){
    cat("\nLineal aproximations of the variables effects:\n")
    print(x$variableCoef, ...)
  }
  if (!is.null(x$predictions)){
    cat("\nPredictions for «predInput» data:\n")
    print(x$predictions, ...)
  }
  if (!is.null(x$model)){
    cat("Models saved in the results.\n")
  }
  if (!is.null(x$variableResponse)){
    cat("Variable response from ingredients::partial_dependency available in the results.\n")
  }
  if (!is.null(x$DALEXexplainer)){
    cat("DALEXexplainers saved in the results.\n")
  }
  invisible(x)
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

build_modelLTSM<- function(input_shape.ts, input_shape.static=0, output_shape=1,
                           hidden_shape.RNN=32, hidden_shape.static=16, hidden_shape.main=32, mask=NULL){
  inputs.ts<- layer_input(shape=input_shape.ts, name="TS_input")
  inputs.static<- layer_input(shape=input_shape.static, name="Static_input")

  if (is.null(mask)){
    predictions.ts<- inputs.ts
    predictions.static<- inputs.static
  } else {
    predictions.ts<- inputs.ts %>% layer_masking(mask_value=mask, input_shape=input_shape.ts, name=paste0("mask.ts_", mask))
    predictions.static<- inputs.static %>% layer_masking(mask_value=mask, input_shape=input_shape.ts, name=paste0("mask.static_", mask))
  }

  predictions.ts<- inputs.ts
  for (i in 1:length(hidden_shape.RNN)){
    predictions.ts<- predictions.ts %>% layer_lstm(units=hidden_shape.RNN[i], name=paste0("LSTM_", i))
  }

  if (input_shape.static > 0){
    predictions.static<- inputs.static
    for (i in 1:length(hidden_shape.static)){
      predictions.static<- predictions.static %>% layer_dense(units=hidden_shape.static[i], name=paste0("Dense_", i))
    }
    output<- layer_concatenate(c(predictions.ts, predictions.static))
  } else {
    output<- predictions.ts
  }

  for (i in 1:length(hidden_shape.main)){
    output<- output %>% layer_dense(units=hidden_shape.main[i], name=paste0("main_dense_", i))
  }
  output<- output %>% layer_dense(units=output_shape, name="main_output")

  if (input_shape.static > 0){
    model<- keras_model(inputs=c(inputs.ts, inputs.static), outputs=output)
  } else {
    model<- keras_model(inputs=inputs.ts, outputs=output)
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
predict_keras<- function(modelNN, predInput, maskNA=NULL, scaleInput=FALSE, col_means_train, col_stddevs_train, batch_size=NULL, filename="", tempdirRaster=NULL, nCoresRaster=2){
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
      predInputScaled<- scale(predInput[, names(col_means_train), drop=FALSE],
                              center=col_means_train, scale=col_stddevs_train)
      predInputScaled<- cbind(predInputScaled, predInput[, setdiff(colnames(predInput), colnames(predInputScaled))])
      if (!is.null(maskNA)){
        predInputScaled<- apply(predInputScaled, 2, function(x){
          x[is.na(x)]<- maskNA
          x
        })
      }
    } else {
      predInputScaled<- predInput
    }

    predicts<- stats::predict(modelNN, predInputScaled, batch_size=batch_size) # TODO, verbose=verbose)
  } else {
    predicts<- stats::predict(modelNN, predInput, batch_size=batch_size) # TODO, verbose=verbose)
  }

  return(predicts)
}
