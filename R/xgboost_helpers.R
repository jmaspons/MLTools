
# @importFrom caret postResample sample_weight
performance_xgboost<- function(modelXGB, test_data, test_labels, sample_weight=NULL, verbose=0){
  # modelValidate <- xgboost::xgb.train(data = testset, nrounds = 0, xgb_model = modelXGB)
  # perf<- modelValidate[nrow(modelValidate$evaluation_log), ]
  # TODO: sample_weight not used
  perfCaret<- caret::postResample(pred=stats::predict(modelXGB, test_data), obs=test_labels) # No weighting

  out<- data.frame(as.list(perfCaret))
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
#' @param ...
#'
#' @return
#' @export
#'
#' @examples
variableImportance<- function(model, data, y, repVi=5, variable_groups=NULL, perm_dim=NULL, comb_dims=FALSE, ...){
  if (repVi > 0){
    vi<- feature_importance(x=model, data=data, y=y, B=repVi, variable_groups=variable_groups,
                                         perm_dim=perm_dim, comb_dims=comb_dims, predict_function=stats::predict, ...)
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

## TODO: move to generic helpers and also remove from keras_helpers.R
variableResponse_explainer<- function(explainer, variables=NULL, maxPoly=5){
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


gatherResults.pipe_result.xgboost<- function(res, aggregate_shap, summarizePred, filenameRasterPred, nCoresRaster, repNames){
  if (missing(repNames) && is.null(repNames)){
    names(res)<- paste0("rep", formatC(1:length(res), format="d", flag="0", width=nchar(length(res))))
  } else {
    names(res)<- repNames
  }

  out<- list(performance=do.call(rbind, lapply(res, function(x) x$performance)))
  sc<- lapply(res, function(x) {
    s<- x$scaleVals
  })
  sc<- sc[!sapply(sc, is.null)]
  if (length(sc) == 1 & all(names(sc[[1]]) == "dataset")){
    sc<- sc[[1]]
  }
  out$scale<- sc

  if (!is.null(res[[1]]$shap)){
    out$shap<- lapply(res, function(x){
      x$shap
    })
    if (aggregate_shap){
      if (all(sapply(out$shap, class) == "shapviz")) {
        out$shap<- do.call(rbind, out$shap)
      }
    }
  }

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
        # Some adj.r.squared & r.squared are NA
        y <- y[x, , drop=TRUE]
        c(y["intercept"], stats::na.omit(y[grep("^b[0-9]+$", names(y))]), y[c("adj.r.squared", "r.squared", "degree")])

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
                dimnames=list(names(varCoef), colNames)) ## TODO: check translation response var from ingredients::partial_dependency()$`_label_`
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
      out$predictions<- do.call(cbind, out$predictions)

      if (summarizePred){
        out$predictions<- summarize_pred.default(out$predictions)
      }
    }

  }


  if (!is.null(res[[1]]$model)){
    out$model<- lapply(res, function(x){
      x$model
    })
  }

  if (!is.null(res[[1]]$explainer)){
    out$DALEXexplainer<- lapply(res, function(x){
      x$explainer
    })
  }

  class(out)<- c("pipe_result.xgboost", "pipe_result")

  return(out)
}

# https://github.com/rspatial/raster/blob/b1c9d91b1b43b17ea757889dc93f97bd70dc1d2e/R/predict.R
# ?raster::`predict,Raster-method`
predict.Raster_xgboost<- function(object, model, filename="", fun=predict, ...) {
  nLayersOut<- 1 # TODO: xgboost in the future will support multiouput models
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

      out<- raster::writeValues(out, matrix(v), bs$row[i])
      raster::pbStep(pb, i)
    }

    out<- raster::writeStop(out)
  } else {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- fun(object=model, v, ...)

      cols<- bs$row[i]:(bs$row[i] + bs$nrows[i] - 1)
      vv[, cols, ]<- array(matrix(v), dim=c(ncol(object), length(cols), nLayersOut))

      raster::pbStep(pb, i)
    }

    out<- raster::setValues(out, as.vector(vv))
  }

  raster::pbClose(pb)

  return(out)
}


#' Predict with xgboost
#'
#' @inheritParams pipe_xgboost
#' @param modelXGB an [xgboost::xgboost()] model.
#' @param predInput `data.frame` or `raster` with colnames or layer names matching the expected input for modelRF.
#' @param scaleInput if `TRUE`, scale `predInput` with `col_means_train` and col `col_stddevs_train`.
#' @param col_means_train the original mean of the `predInput` columns.
#' @param col_stddevs_train the original sd of the `predInput` columns.
#' @param filename the file to write the raster predictions.
#'
#' @return
#'
#' @examples
predict_xgboost<- function(modelXGB, predInput, scaleInput=FALSE, col_means_train, col_stddevs_train, filename="", tempdirRaster=NULL, nCoresRaster=2){
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
      ## TODO: onehot for categorical vars ----
      # predInputScaled<- raster::scale(predInput, center=col_means_train, scale=col_stddevs_train)
      raster::beginCluster(n=nCoresRaster)
      predInputScaled<- raster::clusterR(predInput, function(x, col_means_train, col_stddevs_train){
                                    raster::calc(x, fun=function(y) scale(y, center=col_means_train, scale=col_stddevs_train))
                                  }, args=list(col_means_train=col_means_train, col_stddevs_train=col_stddevs_train), filename=filenameScaled)
      raster::endCluster()
      names(predInputScaled)<- names(predInput)
      # ?raster::`predict,Raster-method`
    } else {
      predInputScaled<- predInput
    }

    predicts<- predict.Raster_xgboost(object=predInputScaled, model=modelXGB, filename=filename) # TODO, verbose=verbose)

    if (scaleInput){
      file.remove(filenameScaled, gsub("\\.grd$", ".gri", filenameScaled))
      rm(predInputScaled)
    }

    # predicti<- raster::predict(object=predInputScaled, model=modelXGB,
    #                    fun=predict,
    #                    filename=filename,
    #                    verbose=verbose)
    # predicts[[i]]<- predicti

  } else if (inherits(predInput, c("data.frame", "matrix"))) {
    if (scaleInput){
      predInputScaled<- scale(predInput[, names(col_means_train), drop=FALSE],
                              center=col_means_train, scale=col_stddevs_train)
      predInputScaled<- cbind(predInputScaled, predInput[, setdiff(colnames(predInput), colnames(predInputScaled))])
    } else {
      predInputScaled<- predInput
    }

    predicts<- stats::predict(modelXGB, predInputScaled) # TODO, verbose=verbose)
  } else {
    predicts<- stats::predict(modelXGB, predInput) # TODO, verbose=verbose)
  }

  return(predicts)
}
