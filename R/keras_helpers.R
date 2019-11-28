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

