#' Random forest model with randomForest
#'
#' @param df a `data.frame` with the data.
#' @param predInput a `data.frame` or a `Raster` with the input variables for the model as columns or layers. The columns or layer names must match the names of `df` columns.
#' @param responseVar response variable as column name or index on `df`.
#' @param caseClass class of the samples used to weight cases. Column names or indexes on `df`, or a vector with the class for each rows in `df`.
#' @param idVars id column names or indexes on `df`. This columns will not be used for training.
#' @param weight Optional array of the same length as `nrow(df)`, containing weights to apply to the model's loss for each sample.
#' @param crossValStrategy `Kfold` or `bootstrap`.
#' @param replicates number of replicates for `crossValStrategy="bootstrap"` and `crossValStrategy="Kfold"` (`replicates * k-1`, 1 fold for validation).
#' @param k number of data partitions when `crossValStrategy="Kfold"`.
#' @param crossValRatio Proportion of the dataset used to train, test and validate the model when `crossValStrategy="bootstrap"`. Default to `c(train=0.6, test=0.2, validate=0.2)`. If there is only one value, will be taken as a train proportion and the test set will be used for validation.
#' @param ntree Number of trees to grow.
#' @param importance parameter for [randomForest::randomForest()] indicating if importance of predictors should be assessed.
#' @param shap if `TRUE`, return the SHAP values as [shapviz::shapviz()] objects.
#' @param aggregate_shap if `TRUE`, and `shap` is also `TRUE`, aggregate SHAP from all replicates.
#' @param repVi replicates of the permutations to calculate the importance of the variables. 0 to avoid calculating variable importance.
#' @param summarizePred if `TRUE`, return the mean, sd and se of the predictors. if `FALSE`, return the predictions for each replicate.
#' @param scaleDataset if `TRUE`, scale the whole dataset only once instead of the train set at each replicate. Optimize processing time for predictions with large rasters.
#' @param RFmodel if TRUE, return the model with the result.
#' @param DALEXexplainer if `TRUE`, return a explainer for the models from [DALEX::explain()] function. It doesn't work with multisession future plans.
#' @param variableResponse if `TRUE`, return aggregated_profiles_explainer object from [ingredients::partial_dependency()] and the coefficients of the adjusted linear model.
#' @param save_validateset save the validateset (independent data not used for training).
#' @param filenameRasterPred if no missing, save the predictions in a RasterBrick to this file.
#' @param tempdirRaster path to a directory to save temporal raster files.
#' @param nCoresRaster number of cores used for parallelized raster cores. Use half of the available cores by default.
#' @param verbose If > 0, print state and passed to randomForest functions
#' @param ... extra parameters for [randomForest::randomForest()], [future.apply::future_replicate()] and [ingredients::feature_importance()].
#'
#' @return
#' @export
# @import randomForest
#' @importFrom stats predict
#' @examples
pipe_randomForest<- function(df, predInput=NULL, responseVar=1, caseClass=NULL, idVars=character(), weight="class",
                   crossValStrategy=c("Kfold", "bootstrap"), k=5, replicates=10, crossValRatio=c(train=0.6, test=0.2, validate=0.2),
                   ntree=500, importance=TRUE,
                   shap=TRUE, aggregate_shap= TRUE, repVi=5, summarizePred=TRUE, scaleDataset=FALSE, RFmodel=FALSE, DALEXexplainer=FALSE, variableResponse=FALSE, save_validateset=FALSE,
                   filenameRasterPred=NULL, tempdirRaster=NULL, nCoresRaster=parallel::detectCores() %/% 2, verbose=0, ...){
  crossValStrategy<- match.arg(crossValStrategy)
  if (length(responseVar) > 1){
    stop("randomForest supports only 1 response variable. Correct `responseVar`")
  }
  if (is.numeric(responseVar)){
    responseVar<- colnames(df)[responseVar]
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

  predVars<- setdiff(colnames(df), c(responseVar, idVars))
  predVars.cat<- names(which(!sapply(df[, predVars, drop=FALSE], is.numeric)))
  predVars.num<- setdiff(predVars, predVars.cat)

  # Need to load caret https://github.com/topepo/caret/issues/380
  ## df[, predVars.cat]<- lapply(df[, predVars.cat], factor)
  # catEnc<- caret::dummyVars(stats::as.formula(paste("~", paste(predVars.cat, collapse="+"))), data=df)
  # caret:::predict.dummyVars(catEnc, newdata=df)
  # stats::model.matrix(~0+df[, predVars.cat])
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

    rm(df.catBin)
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

        rm(predInput.catBin)
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
        predInputScaled<- raster::clusterR(predInput[[names(col_means_train)]], function(x, col_means_train, col_stddevs_train){
                              raster::calc(x, fun=function(y) scale(y, center=col_means_train, scale=col_stddevs_train))
                            }, args=list(col_means_train=col_means_train, col_stddevs_train=col_stddevs_train), filename=filenameScaled)
        predInput<- predInputScaled
        names(predInput)<- names(col_means_train)
        raster::endCluster()

      }  else if (inherits(predInput, c("data.frame", "matrix"))) {
        predInput[, names(col_means_train)]<- scale(predInput[, names(col_means_train), drop=FALSE], center=col_means_train, scale=col_stddevs_train)
      }

    }
  }


  idxSetsL<- switch(crossValStrategy,
    bootstrap=bootstrap_train_test_validate(df, replicates=replicates, ratio=crossValRatio, caseClass=caseClass, weight=weight),
    Kfold=kFold_train_test_validate(d=df, k=k, replicates=replicates, caseClass=caseClass, weight=weight)
  )

  res<- future.apply::future_lapply(idxSetsL, function(idx.repli){
  # DEBUG: idx.repli<- idxSetsL[[1]]
  # res<- list()
  # for (i in seq_along(idxSetsL)){
  #   idx.repli<- idxSetsL[[i]]
    # TODO: TEST idx.repli<- idxSetsL[[1]]
    resi<- list()
    # crossValSets<- splitdf(df, ratio=crossValRatio, sample_weight=sample_weight)
    crossValSets<- lapply(idx.repli[intersect(c("trainset", "testset"), names(idx.repli))], function(x) df[x, ])

    train_labels<- crossValSets$trainset[, responseVar, drop=FALSE]
    train_data<- crossValSets$trainset[, predVars, drop=FALSE]

    test_labels<- crossValSets$testset[, responseVar, drop=FALSE]
    test_data<- crossValSets$testset[, predVars, drop=FALSE]

    sample_weight<- idx.repli[intersect(c("weight.train", "weight.test"), names(idx.repli))]
    # if (length(sample_weight) == 0) sample_weight<- NULL

    # If no validation set exist, use test set to check performance
    if (length(idx.repli$validateset) > 0){
      validate_labels<- df[idx.repli$validateset, responseVar, drop=FALSE]
      validate_data<- df[idx.repli$validateset, predVars, drop=FALSE]
      if (!is.null(idx.repli$weight.validate)){
        sample_weight$weight.validate<- idx.repli$weight.validate
      }
    } else {
      validate_labels<- test_labels
      validate_data<- test_data

      if (!is.null(idx.repli$weight.test)){
        sample_weight$weight.validate<- sample_weight$weight.test
      }
    }

    if (!scaleDataset){
      train_data.scaled<- scale(train_data[, predVars.num, drop=FALSE])
      train_data[, predVars.num]<- train_data.scaled

      col_means_train<- attr(train_data.scaled, "scaled:center")
      col_stddevs_train<- attr(train_data.scaled, "scaled:scale")
      rm(train_data.scaled)

      test_data[, names(col_means_train)]<- scale(test_data[, names(col_means_train), drop=FALSE], center=col_means_train, scale=col_stddevs_train)
      validate_data[, names(col_means_train)]<- scale(validate_data[, names(col_means_train), drop=FALSE], center=col_means_train, scale=col_stddevs_train)

      resi$scaleVals<- data.frame(mean=col_means_train, sd=col_stddevs_train)
    }

    ## Not needed for randomForest but for caret & DALEX(?)
    # if (inherits(train_data, "data.frame")){
    #   train_data<- as.matrix(train_data)
    # }
    # if (inherits(test_data, "data.frame")){
    #   test_data<- as.matrix(test_data)
    # }
    # if (inherits(validate_data, "data.frame")){
    #   validate_data<- as.matrix(validate_data)
    # }
    #
    # if (inherits(train_labels, "data.frame")){
    #   train_labels<- as.matrix(train_labels)
    # }
    # if (inherits(test_labels, "data.frame")){
    #   test_labels<- as.matrix(test_labels)
    # }
    # if (inherits(validate_labels, "data.frame")){
    #   validate_labels<- as.matrix(validate_labels)
    # }

    # modelRF<- randomForest::randomForest(x=train_data, y=train_labels[[1]], xtest=test_data, ytest=test_labels[[1]],
    #                                      ntree=ntree, weights=sample_weight$weight.train,
    #                                      na.action=na.omit, importance=importance,
    #                                      do.trace=ifelse(verbose > 2, TRUE, FALSE), keep.forest=RFmodel, ...)
    # TODO: use formula to deal with NAs
    form<- stats::as.formula(paste(colnames(train_labels), "~ ."))
    modelRF<- randomForest::randomForest(formula=form, data=cbind(train_data, train_labels), xtest=test_data, ytest=test_labels[, 1],
                                         ntree=ntree, weights=sample_weight$weight.train, na.action=stats::na.omit, importance=importance,
                                         do.trace=ifelse(verbose > 2, TRUE, FALSE), keep.forest=TRUE, ...)
    if (verbose > 1) message("Training done")

    if (RFmodel){
      resi$model<- modelRF
    }

    ## Model performance
    if (length(sample_weight) > 0){
      sample_weight.validate<- as.matrix(sample_weight$weight.validate)
    } else {
      sample_weight.validate<- NULL
    }
    resi$performance<- performance_randomForest(modelRF=modelRF, test_data=validate_data, test_labels=validate_labels[, 1],
                                                sample_weight=sample_weight.validate, verbose=max(c(0, verbose - 2)))

    if (verbose > 1) message("Performance analyses done")

    ## SHAP
    if (shap){
      resi$shap<- get_shap(model=modelRF, test_data=validate_data, bg_data=train_data, bg_weight=sample_weight$weight.train, verbose=max(c(0, verbose - 2)))
    }

    ## Variable importance
    if (repVi > 0){
      variable_groups<- NULL
      if (length(predVars.cat) > 0){
        # Join variable importance for predictors from the same categorical variable
        variable_groups<- lapply(predVars.cat, function(x){
          grep(paste0("^", x), colnames(validate_data), value=TRUE)
        })
        variable_groups<- stats::setNames(variable_groups, nm=predVars.cat)
        predVarsNumOri<- setdiff(predVars, predVars.catBin)
        variable_groups<- c(stats::setNames(as.list(predVarsNumOri), nm=predVarsNumOri), variable_groups)
      }
      resi$variableImportance<- variableImportance(model=modelRF, data=validate_data, y=validate_labels[[1]], repVi=repVi,
                                                   variable_groups=variable_groups, ...)
    }

    ## Explain model
    if (variableResponse | DALEXexplainer){
      explainer<- DALEX::explain(model=modelRF, data=validate_data, y=as.matrix(validate_labels), predict_function=stats::predict,
                                 weights=sample_weight.validate, verbose=FALSE)

      ## Variable response
      if (variableResponse){
        resi$variableResponse<- variableResponse_explainer(explainer)
      }

      if (DALEXexplainer){
        resi$explainer<- explainer
      }
    }
    if (verbose > 1) message("Model analyses done")

    ## Predictions
    if (!is.null(predInput)){
      resi$predictions<- predict_randomForest(modelRF=modelRF, predInput=predInput, scaleInput=!scaleDataset,
                             col_means_train=col_means_train, col_stddevs_train=col_stddevs_train,
                             tempdirRaster=tempdirRaster, nCoresRaster=nCoresRaster)
      if (inherits(resi$predictions, "matrix")){
        colnames(resi$predictions)<- responseVar
        rownames(resi$predictions)<- rownames(predInput)
      } else if (inherits(resi$predictions, "Raster")){
        names(resi$predictions)<- responseVar
      }
    }
    if (verbose > 1) message("Prediction done")

    return(resi)
  }, future.seed=TRUE, ...)
  #   res[[i]]<- resi
  # }

  if (scaleDataset){
    res[[1]]$scaleVals<- list(dataset=data.frame(mean=col_means_train, sd=col_stddevs_train))
    if (inherits(predInput, "Raster")){
      file.remove(filenameScaled, gsub("\\.grd$", ".gri", filenameScaled))
    }
  }

  if (verbose > 0) message("Iterations finished. Gathering results...")

  out<- gatherResults.pipe_result.randomForest(res=res, aggregate_shap=aggregate_shap, summarizePred=summarizePred, filenameRasterPred=filenameRasterPred, nCoresRaster=nCoresRaster, repNames=names(idxSetsL))

  out$params<- list(responseVar=responseVar, predVars=predVars, predVars.cat=predVars.cat,
                    caseClass=caseClass, idVars=idVars, weight=weight,
                    crossValStrategy=crossValStrategy, k=k, replicates=replicates, crossValRatio=crossValRatio,
                    importance=importance, ntree=ntree,
                    shap=shap, aggregate_shap=aggregate_shap, repVi=repVi, summarizePred=summarizePred, scaleDataset=scaleDataset, RFmodel=RFmodel, DALEXexplainer=DALEXexplainer, variableResponse=variableResponse,
                    save_validateset=save_validateset, filenameRasterPred=filenameRasterPred)
  if (crossValStrategy != "Kfold") out$params$k<- NULL


  if (!is.null(predInput) & inherits(predInput, c("data.frame", "matrix")) & length(idVarsPred) > 0){
    ## Add idVars if exists
    out$predictions<- cbind(predInputIdVars, out$predictions)
  }

  if (save_validateset){
    out$validateset<- lapply(idxSetsL, function(x) df[x$validateset, ])
  }

  class(out)<- c("pipe_result.randomForest", "pipe_result")

  return(out)
}
