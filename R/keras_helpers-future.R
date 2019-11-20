#' Title
#'
#' @param df a data.frame whith response variable in the first column
#' @param predInput a Raster or a data.frame with columns 1 and 2 corresponding to longitude and latitude + variables for the model
#' @param idVars id column names or indexes on df and/or predInput. Deprecated default for compatibility c("x", "y")
#' @param responseVars response variables. Column names or indexes on df
#' @param epochs parameter for \code\link[keras]{fit}}.
#' @param replicates number of replicates
#' @param repVi replicates of the permutations to calculate the importance of the variables. 0 to avoid report variable importance
#' @param DALEXexplainer return a explainer for the models from \code\link[DALEX]{explain}} function.
#' @param crossValRatio Proportion of the dataset used to train the model. Default to 0.8
#' @param NNmodel if TRUE, return the serialized model with the result.
#' @param baseFilenameNN if no missing, save the NN in hdf5 format on this path with iteration appended.
#' @param batch_size for fit and predict functions. The bigger the better if it fits your available memory. Integer or "all".
#' @param baseFilenameRasterPred if no missing, save the predictions in Raster format on this path with iteration appended.
#' @param verbose If > 0, print state and passed to keras functions
#'
#' @return
#' @export
#' @import keras
#' @importFrom caret postResample
#' @importFrom DALEX explain
#' @importFrom ingredients feature_importance
#' @importFrom stats predict
#' @examples
process_keras<- function(df, predInput, responseVars=1, idVars=character(),
                   epochs=500, replicates=10, repVi=5, DALEXexplainer=TRUE, crossValRatio=0.8,
                   NNmodel=TRUE, baseFilenameNN, batch_size="all", baseFilenameRasterPred, verbose=0){
  if (is.character(responseVars)){
    responseVars<- which(colnames(df) %in% responseVars)
  }
  if (is.character(idVars)){
    idVars<- which(colnames(df) %in% idVars)
  }
  predVars<- setdiff(1:ncol(df), c(responseVars, idVars))

  if (missing(predInput)) predInput<- NULL
  if (missing(baseFilenameNN)) baseFilenameNN<- NULL
  if (missing(baseFilenameRasterPred)) baseFilenameRasterPred<- NULL
  # verbose<- verbose ## for future?

  modelNN<- build_modelDNN(input_shape=length(predVars), output_shape=length(responseVars))

  ## Check convergence on the max epochs frame
  early_stop<- keras::callback_early_stopping(monitor="val_loss", patience=30)

  res<- future.apply::future_replicate(replicates, {
    resi<- list()
    crossValSets<- NNTools:::splitdf(df, ratio=crossValRatio)

    train_labels<- as.matrix(crossValSets$trainset[, responseVars, drop=FALSE])
    train_data<- as.matrix(crossValSets$trainset[, predVars, drop=FALSE])

    test_labels<- as.matrix(crossValSets$testset[, responseVars, drop=FALSE])
    test_data<- as.matrix(crossValSets$testset[, predVars, drop=FALSE])

    train_data<- scale(train_data)

    col_means_train<- attr(train_data, "scaled:center")
    col_stddevs_train<- attr(train_data, "scaled:scale")

    resi$scaleVals<- data.frame(mean=col_means_train, sd=col_stddevs_train)

    test_data<- scale(test_data, center=col_means_train, scale=col_stddevs_train)

    ## TODO: check if reset_state is faster and equivalent to build_model
    # modelNN<- build_modelDNN(input_shape=length(predVars), output_shape=length(responseVars))
    modelNN<- keras::reset_states(modelNN)


    resi$model<- NNTools:::train_keras(modelNN=modelNN, train_data=train_data, train_labels=train_labels,
                           test_data=test_data, test_labels=test_labels, epochs=epochs,
                           batch_size=batch_size, callbacks=early_stop, verbose=verbose)

    ## Model performance
    resi$performance<- NNTools:::performance_keras(modelNN=modelNN, test_data=test_data, test_labels=test_labels,
                             batch_size=ifelse(batch_size %in% "all", nrow(test_data), batch_size), verbose=verbose)

    ## Variable importance
    resi$variableImportance<- NNTools:::variableImportance_keras(modelNN=resi$model, train_data=train_data, train_labels=train_labels,
                                     repVi=repVi, DALEXexplainer=DALEXexplainer)

    ## Predictions
    if (!is.null(predInput)){
      if (inherits(predInput, "Raster")){
        batch_sizePred<- ifelse(batch_size %in% "all", raster::ncell(predInput), batch_size)
        if (!is.null(baseFilenameRasterPred)){
          filename<- paste0(baseFilenameRasterPred, "_it", formatC(i, format="d", flag="0", width=nchar(replicates)), ".grd")
        }else{
          filename<- ""
        }
      } else {
        batch_sizePred<- ifelse(batch_size %in% "all", nrow(predInput), batch_size)
      }

      resi$predictions<- NNTools:::predict_keras(modelNN=modelNN, predInput=predInput,
                               col_means_train=col_means_train, col_stddevs_train=col_stddevs_train,
                               batch_size=batch_sizePred, filename=filename)
    }

    return(resi)
  }, simplify=FALSE)



  if (!is.null(baseFilenameNN)){
    out<- lapply(seq_along(res), function(i){
      save_model_hdf5(res[[i]]$model, filepath=paste0(baseFilenameNN, "_", formatC(i, format="d", flag="0", width=nchar(replicates)), ".hdf5"),
                      overwrite=TRUE, include_optimizer=TRUE)
    })
  }


  out<- list(performance=do.call(rbind, lapply(res, function(x) x$performance)),
             scale=lapply(res, function(x) x$scaleVals))


  if (repVi > 0){
    vi<- lapply(res, function(x){
            tmp<- x$variableImportance$vi
            tmp[sort(rownames(tmp)),]
          })
    vi<- do.call(cbind, vi)
    out$vi<- vi[order(rowSums(vi)), ] ## Order by average vi
  }

  if (!is.null(predInput)){
    out$predictions<- lapply(res, function(x) x$predictions)
## TODO: pack rasters
    if (inherits(predInput, "Raster")){
      if (!all(sapply(out$predictions, raster::inMemory))){
        warning("The rasters with the predictions doesn't fit in memory and the values are saved in a temporal file. ",
                "Please, provide the baseFilenameRasterPred parameter to save the raster in a non temporal file. ",
                "If you want to save the predictions of the current run use writeRaster on result$predicts before to close the session.")
      }
    } else {
      out$predictions<- do.call(cbind, out$predictions)
      rownames(out$predictions)<- rownames(predInput)
      colnames(out$predictions)<- paste0(rep(colnames(df)[responseVars], times=replicates),
               rep(paste0("_rep", formatC(1:replicates, format="d", flag="0", width=nchar(replicates))),
                   each=length(responseVars)))

      idVarNames<- intersect(colnames(df)[idVars], colnames(predInput))

      if (length(idVarNames) > 0){
        out$predictions<- cbind(predInput[, idVarNames, drop=FALSE], out$predictions)
      }

    }
  }


  if (NNmodel){
    out$model<- lapply(res, function(x){
      serialize_model(x$model) # unserialize_model() to use the saved model
    })
  }

  if (DALEXexplainer){
    out$DALEXexplainer<- lapply(res, function(x){
      x$variableImportance$explainer
    })
  }

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


performance_keras<- function(modelNN, test_data, test_labels, batch_size, verbose=0){
  perf<- keras::evaluate(modelNN, test_data, test_labels, verbose=verbose,
                  batch_size=batch_size)
  perfCaret<- caret::postResample(pred=stats::predict(modelNN, test_data), obs=test_labels)

  out<- data.frame(data.frame(perf), as.list(perfCaret))
  return(out)
}


variableImportance_keras<- function(modelNN, train_data, train_labels, repVi=5, DALEXexplainer=FALSE){
  out<- list()

  if (repVi > 0 | DALEXexplainer){
    explainer<- DALEX::explain(model=modelNN, data=train_data, y=train_labels, predict_function=stats::predict, label="MLP_keras", verbose=FALSE)
  }

  if (repVi > 0){
    vi<- replicate(n=repVi, ingredients::feature_importance(explainer), simplify=FALSE)
    vi<- structure(sapply(vi, function(x) x$dropout_loss),
                   dimnames=list(as.character(vi[[1]]$variable), paste0("rep", 1:repVi)))
    out$vi<- vi
  }

  if (DALEXexplainer){
    out$explainer<- explainer
  }

  return(out)
}


predict_keras<- function(modelNN, predInput, col_means_train, col_stddevs_train, batch_size, filename=""){
  if (inherits(predInput, "Raster") & requireNamespace("raster", quietly=TRUE)){
    selCols<- intersect(names(predInput), names(col_means_train))

    predInputScaled<- raster::scale(predInput[[selCols]],
                                    center=col_means_train[selCols], scale=col_stddevs_train[selCols])

    # ?keras::predict.keras.engine.training.Model
    # ?raster::`predict,Raster-method`

    predicts<- predict.Raster_keras(object=predInputScaled, model=modelNN, filename=filename,
                                    batch_size=batch_size) # TODO, verbose=verbose)

    # predicti<- raster::predict(object=predInputScaled, model=modelNN,
    #                    fun=keras:::predict.keras.engine.training.Model,
    #                    filename=filename,
    #                    batch_size=ifelse(batch_size %in% "all", raster::ncell(predInputScaled), batch_size),
    #                    verbose=verbose)
    # predicts[[i]]<- predicti

  } else if (inherits(predInput, c("data.frame", "matrix"))) {
    selCols<- intersect(colnames(predInput), names(col_means_train))

    predInputScaled<- scale(predInput[, selCols, drop=FALSE],
                            center=col_means_train[selCols], scale=col_stddevs_train[selCols])
    predicts<- stats::predict(modelNN, predInputScaled,
                              batch_size=batch_size) # TODO, verbose=verbose)
  }

  return(predicts)
}
