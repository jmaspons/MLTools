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


#' Title
#'
#' @param df a data.frame whith response variable in the first column
#' @param predInput a Raster or a data.frame with columns 1 and 2 corresponding to longitude and latitude + variables for the model
#' @param idVars id column names or indexes on df and/or predInput. Deprecated default for compatibility c("x", "y")
#' @param responseVars response variables. Column names or indexes on df
#' @param epochs parameter for \code\link[keras]{fit}}.
#' @param iterations number of replicates
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
process<- function(df, predInput, responseVars=1, idVars=character(),
                   epochs=500, iterations=10, repVi=5, DALEXexplainer=TRUE, crossValRatio=0.8,
                   NNmodel=TRUE, baseFilenameNN, batch_size="all", baseFilenameRasterPred, verbose=0){
  .Deprecated("process_keras", package="NNTools")
  perf<- data.frame()
  scaleVals<- list()
  DALEXexplainerL<- list()
  NNmodelL<- list()

  if (is.character(responseVars)){
    responseVars<- which(colnames(df) %in% responseVars)
  }
  if (is.character(idVars)){
    idVars<- which(colnames(df) %in% idVars)
  }
  predVars<- setdiff(1:ncol(df), c(responseVars, idVars))

  if (!missing(predInput)){
    if (inherits(predInput, "Raster")){
      predRaster<- TRUE

      predicts<- list()
    }else{
      predRaster<- FALSE
      predicts<- matrix(ncol=length(idVars) + (iterations * length(responseVars)), nrow=nrow(predInput),
                        dimnames=list(rownames(predInput),
                          c(colnames(df)[idVars], paste0(rep(colnames(df)[responseVars], times=iterations), rep(paste0("_rep", formatC(1:iterations, format="d", flag="0", width=nchar(iterations))), each=length(responseVars))))))
      predicts<- data.frame(predicts, stringsAsFactors=FALSE)
      predicts[, idVars]<- predInput[, idVars]
    }
  }

  modelNN<- build_modelDNN(input_shape=length(predVars), output_shape=length(responseVars))

  # pb<- txtProgressBar(max=iterations, style=pbStyle)
  # on.exit(close(pb))
  # pbapply::pboptions(style=3)
  pb<- pbapply::startpb(min=1, max=iterations)
  on.exit(pbapply::closepb(pb))
  for (i in 1:iterations) {
    ## TODO: use replicate(n=iterations, expr, simplify=FALSE) or
    # batchtools to fix exponential process time increase
    # setTxtProgressBar(pb, i)
    pbapply::setpb(pb, i)
    if (verbose > 0) message("\t", i, " / ", iterations, "\t", appendLF=FALSE)
    crossValSets<- splitdf(df, ratio=crossValRatio)

    trainY<- as.matrix(crossValSets$trainset[, responseVars])
    trainX<- as.matrix(crossValSets$trainset[, predVars])
    trainset<- list(trainX, trainY)

    testY<- as.matrix(crossValSets$testset[, responseVars])
    testX<- as.matrix(crossValSets$testset[, predVars])
    testset<- list(testX,testY)

    Tr1Nnn<- list(trainset, testset)
    ## TODO: translate to R
    c(train_data, train_labels) %<-% Tr1Nnn[[1]]
    c(test_data, test_labels) %<-% Tr1Nnn[[2]]
    names(Tr1Nnn)[[1]]<- c(train_data, train_labels)

    train_data<- scale(train_data)
    # train_df<- as_tibble(train_data)

    col_means_train<- attr(train_data, "scaled:center")
    col_stddevs_train<- attr(train_data, "scaled:scale")

    scaleVals<- c(scaleVals, list(data.frame(mean=col_means_train, sd=col_stddevs_train)))


    test_data<- scale(test_data, center=col_means_train, scale=col_stddevs_train)

    ## TODO: check if reset_state is faster and equivalent to build_model
    # modelNN<- build_model(train_data, train_labels)
    modelNN<- reset_states(modelNN)

    # print_dot_callback<- callback_lambda(
    #   on_epoch_end = function(epoch, logs) {
    #     if (epoch %% 80 == 0) message("")
    #     message(".", appendLF=FALSE)
    #   }
    # )

    # history<- fit(modelNN,
    #   train_data,
    #   train_labels,
    #   epochs = epochs,
    #   validation_split = 0.2,
    #   verbose = 0,
    #   callbacks = list(print_dot_callback)
    # )

    ## Check convergence on the max epochs frame
    early_stop<- callback_early_stopping(monitor="val_loss", patience=30)

    if (verbose > 0) message("trainning... ", appendLF=FALSE)

    history<- fit(modelNN,
      train_data,
      train_labels,
      epochs=epochs,
      validation_data=list(test_data, test_labels),
      verbose=verbose,
      batch_size=ifelse(batch_size %in% "all", nrow(train_data), batch_size),
      callbacks=list(early_stop)#, print_dot_callback)
    )


    if (!missing(baseFilenameNN)){
      save_model_hdf5(modelNN, filepath=paste0(baseFilenameNN, "_", formatC(i, format="d", flag="0", width=nchar(iterations)), ".hdf5"),
                      overwrite=TRUE, include_optimizer=TRUE)
    }
    if (NNmodel){
      NNmodelL[[i]]<- serialize_model(modelNN) # unserialize_model() to use the saved model
    }

    ## Model performance
    if (verbose > 0) message("perfomance... ", appendLF=FALSE)

    perfi<- data.frame(evaluate(modelNN, test_data, test_labels, verbose=verbose, batch_size=ifelse(batch_size %in% "all", nrow(test_data), batch_size)))
    perfCaret<- caret::postResample(pred=stats::predict(modelNN, test_data), obs=test_labels)
    mean<- mean(train_labels)

    perfi<- data.frame(mean=mean, perfi, as.list(perfCaret))
    perf<- rbind(perf, perfi)

    ## Variable importance
    if (verbose > 0 & repVi > 0) message("vi... ", appendLF=FALSE)

    if (repVi > 0 | DALEXexplainer){
      explainer<- DALEX::explain(model=modelNN, data=train_data, y=train_labels, predict_function=stats::predict, label="MLP_keras", verbose=FALSE)
    }
    if (repVi > 0){
      vii<- replicate(n=repVi, ingredients::feature_importance(explainer), simplify=FALSE)
      vii<- structure(sapply(vii, function(x) x$dropout_loss),
                      dimnames=list(as.character(vii[[1]]$variable), paste0("it", i, "_rep", 1:repVi)))

      if (i == 1){
        vi<- vii
      }else{
        vi<- cbind(vi, vii[match(rownames(vii), rownames(vi)), ])
      }
    }

    if (DALEXexplainer){
      DALEXexplainerL[[i]]<- explainer
    }

    ## Predictions
    if (!missing(predInput)){
      if (verbose > 0) message("predict... ", appendLF=FALSE)

      if (predRaster & requireNamespace("raster", quietly=TRUE)){
        selCols<- intersect(names(predInput), names(col_means_train))

        predInputScaled<- raster::scale(predInput[[selCols]],
                                center=col_means_train[selCols], scale=col_stddevs_train[selCols])

        if (!missing(baseFilenameRasterPred)){
          filename<- paste0(baseFilenameRasterPred, "_it", formatC(i, format="d", flag="0", width=nchar(iterations)), ".grd")
        }else{
          filename<- ""
        }

        # ?keras::predict.keras.engine.training.Model
        # ?raster::`predict,Raster-method`

        predicti<- predict.Raster_keras(object=predInputScaled, model=modelNN, filename=filename,
                                        batch_size=ifelse(batch_size %in% "all", raster::ncell(predInputScaled), batch_size),
                                        verbose=verbose)

        # predicti<- raster::predict(object=predInputScaled, model=modelNN,
        #                    fun=keras:::predict.keras.engine.training.Model,
        #                    filename=filename,
        #                    batch_size=ifelse(batch_size %in% "all", raster::ncell(predInputScaled), batch_size),
        #                    verbose=verbose)
        predicts[[i]]<- predicti

      } else if (!predRaster) {
        selCols<- intersect(colnames(predInput), names(col_means_train))

        predInputScaled<- scale(predInput[, selCols],
                                center=col_means_train[selCols], scale=col_stddevs_train[selCols])
        predicti<- stats::predict(modelNN, predInputScaled,
                           batch_size=ifelse(batch_size %in% "all", nrow(predInputScaled), batch_size),
                           verbose=verbose)
        predicts[, (length(idVars) + i * length(responseVars) - (length(responseVars) - 1)):(length(idVars) + i * length(responseVars))]<- predicti
      }
    }
    if (verbose >= 0) message("")
  }


  res<- list(performance=perf, scale=scaleVals)

  if (NNmodel){
    res$NNmodel<- NNmodelL # unserialize_model() to use the saved model
  }
  if (repVi > 0){
    res$vi<- vi
  }
  if (DALEXexplainer){
    res$DALEXexplainer<- DALEXexplainerL
  }
  if (!missing(predInput)){
    res$predicts<- predicts

    if (predRaster)
      if (!all(sapply(predicts, raster::inMemory))){
        warning("The rasters with the predictions doesn't fit in memory and the values are saved in a temporal file. ",
              "Please, provide the baseFilenameRasterPred parameter to save the raster in a non temporal file. ",
              "If you want to save the predictions of the current run use writeRaster on result$predicts before to close the session.")
      }
  }

  return(res)
}


# https://github.com/rspatial/raster/blob/b1c9d91b1b43b17ea757889dc93f97bd70dc1d2e/R/predict.R
# ?raster::`predict,Raster-method`
# ?predict.keras.engine.training.Model
## TODO: length(responseVars) > 1
predict.Raster_keras<- function(object, model, filename, fun=predict, ...) {
  out<- raster::raster(object)
  big<- !raster::canProcessInMemory(out, raster::nlayers(object) + 1)
  filename<- raster::trim(filename)

  if (big & filename == "") {
    filename<- raster::rasterTmpFile()
  }

  if (filename != "") {
    out<- raster::writeStart(out, filename)
    todisk<- TRUE
  } else {
    vv<- matrix(ncol=nrow(out), nrow=ncol(out))
    todisk<- FALSE
  }

  bs<- raster::blockSize(object)
  pb<- raster::pbCreate(bs$n)

  if (todisk) {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- predict(object=model, v, ...)

      out<- raster::writeValues(out, v, bs$row[i])
      raster::pbStep(pb, i)
    }

    out<- raster::writeStop(out)
  } else {
    for (i in 1:bs$n) {
      v<- raster::getValues(object, row=bs$row[i], nrows=bs$nrows[i])
      v<- predict(object=model, v, ...)

      cols<- bs$row[i]:(bs$row[i] + bs$nrows[i] - 1)
      vv[,cols]<- matrix(v, nrow=out@ncols)

      raster::pbStep(pb, i)
    }

    out<- raster::setValues(out, as.vector(vv))
  }

  raster::pbClose(pb)

  return(out)
}

## DEPRECATED ----
build_model<- function(train_data, train_labels=matrix(1)) {
  .Deprecated("build_modelDNN", package="NNTools")
  input_shape<- dim(train_data)[2]
  output_shape<- dim(train_labels)[2]
  model<- keras_model_sequential() %>%
    layer_dense(units=128, activation="relu",
                input_shape=input_shape) %>%
    # layer_dense(units=128, activation="relu") %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dense(units=output_shape)

  model %>% compile(
    loss="mse",
    optimizer=optimizer_rmsprop(),
    metrics=list("mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error")
  )

  model
}

trainPred<- function(df, predInput, epochs=500, repVi=5, filenameNN, batch_size=NULL, verbose=0){
  .Deprecated("process", package="NNTools")
  perf<- data.frame()
  scaleVals<- list()

  # predicts<- data.frame(matrix(ncol=iterations + 2, nrow=nrow(predInput)))
  # predicts[, 1:2]<- predInput[, c("x", "y")]
  # names(predicts)[1:2]<- c("Longitude", "Latitude", paste("rep", 1:iterations))


  # for (i in 1:iterations) {
    # if (verbose >= 0) message("\t", i, " / ", iterations, "\t", appendLF=FALSE)
    # crossValSets<- splitdf(df, ratio=0.8)

    trainY<- as.matrix(df[, 1, drop=FALSE])
    trainX<- as.matrix(df[, -1, drop=FALSE])
    trainset<- list(trainX, trainY)

    # testY<- as.matrix(crossValSets$testset[, 1])
    # testX<- as.matrix(crossValSets$testset[, -1]) # definir les columnes desitjades
    # testset<- list(testX,testY)

    Tr1Nnn<- list(trainset, trainset)
    c(train_data, train_labels) %<-% Tr1Nnn[[1]]
    # c(test_data, test_labels) %<-% Tr1Nnn[[2]]
    names(Tr1Nnn)[[1]]<- c(train_data, train_labels)

    train_data<- scale(train_data)

    col_means_train<- attr(train_data, "scaled:center")
    col_stddevs_train<- attr(train_data, "scaled:scale")

    scaleVals<- c(scaleVals, list(data.frame(mean=col_means_train, sd=col_stddevs_train)))


    # test_data<- scale(test_data, center=col_means_train, scale=col_stddevs_train)

    modelNN<- build_model(train_data, train_labels)

    # print_dot_callback<- callback_lambda(
    #   on_epoch_end = function(epoch, logs) {
    #     if (epoch %% 80 == 0) message("")
    #     message(".", appendLF=FALSE)
    #   }
    # )

    # history<- modelNN %>% fit(
    #   train_data,
    #   train_labels,
    #   epochs = epochs,
    #   validation_split = 0.2,
    #   verbose = 0,
    #   callbacks = list(print_dot_callback)
    # )

    ## Check convergence on the max epochs frame
    early_stop<- callback_early_stopping(monitor="val_loss", patience=30)

    if (verbose >= 0) message("trainning... ", appendLF=FALSE)

    history<- modelNN %>% fit(
      train_data,
      train_labels,
      epochs=epochs,
      validation_data=list(train_data, train_labels),
      verbose=verbose,
      batch_size=ifelse(batch_size %in% "all", nrow(train_data), batch_size),
      callbacks=list(early_stop)#, print_dot_callback)
    )


    if (!missing(filenameNN)){
      save_model_hdf5(modelNN, paste0(filenameNN, ".hdf5"), overwrite=TRUE, include_optimizer=TRUE)
      # save_model_hdf5(modelNN, paste0(baseFilenameNN, "_", formatC(i, format="d", flag="0", width=nchar(iterations)), ".hdf5"),
      #                 overwrite=TRUE, include_optimizer=TRUE)
    }

    ## Model performance
    if (verbose >= 0) message("perfomance... ", appendLF=FALSE)

    perfi<- data.frame(modelNN %>% evaluate(train_data, train_labels, verbose=verbose, batch_size=ifelse(batch_size %in% "all", nrow(train_data), batch_size)))
    perfCaret<- caret::postResample(pred=stats::predict(modelNN, train_data), obs=train_labels)
    mean<- mean(train_labels)

    perfi<- data.frame(mean=mean, perfi, as.list(perfCaret))
    perf<- rbind(perf, perfi)

    ## Variable importance
    if (verbose >= 0 & repVi > 0) message("vi... ", appendLF=FALSE)

    explainer<- DALEX::explain(model=modelNN, data=train_data, y=train_labels, predict_function=predict, label="MLP_keras")
    vii<- replicate(n=repVi, ingredients::feature_importance(explainer), simplify=FALSE)
    vii<- structure(sapply(vii, function(x) x$dropout_loss),
                    dimnames=list(as.character(vii[[1]]$variable), paste0("rep", 1:repVi)))

    # if (i == 1){
      vi<- vii
    # }else{
    #   vi<- cbind(vi, vii[match(rownames(vii), rownames(vi)), ])
    # }

    # if (!missing(predInput)){
      if (verbose >= 0) message("predict... ", appendLF=FALSE)
      selCols<- intersect(colnames(predInput), names(col_means_train))
      # setdiff(colnames(train_data), selCols)
      predInputScaled<- scale(predInput[, selCols],
                              center=col_means_train[selCols], scale=col_stddevs_train[selCols])
      predicts<- predict(modelNN, predInputScaled,
                           batch_size=ifelse(batch_size %in% "all", nrow(predInputScaled), batch_size),
                           verbose=verbose)
    # }
    if (verbose >= 0) message("")
  # }

  res<- list(performance=perf, vi=vi, scale=scaleVals, predicts=predicts)

  return(res)
}


trainPred.map<- function(df, predInput, epochs=500, repVi=5, rasterFile, baseFilenameNN, batch_size=NULL, verbose=0){
  .Deprecated("process", package="NNTools")
  predInputDF<- predInput[]

  res<- trainPred(df=df, predInput=predInputDF, epochs=epochs, repVi=repVi,  baseFilenameNN, batch_size=batch_size, verbose=verbose)

  if (verbose >= 0) message("Creating a raster map\n")

  res$predicts<- raster(res$predicts, template=predInput)

  if (!missing(rasterFile)){
    writeRaster(res$predicts, filename=rasterFile, progress="text")
  }
  return(res)
}
