context("NN_keras")

test_that("Process works", {
  df<- data.frame(replicate(10, runif(100)))
  predInput<- data.frame(replicate(10, runif(50)))

  res<- process(df, predInput=predInput, epochs=10, iterations=5, repVi=2, batch_size="all",
                DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)

  res<- process(df, predInput=predInput, responseVars=1:2, epochs=10, iterations=5, repVi=2, batch_size="all",
                DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)
})

test_that("Predict with raster", {
  df<- data.frame(replicate(10, runif(100)))

  predInput<- raster::raster(nrows=15, ncols=15)
  predInput<- raster::stack(replicate(10, {
    raster::setValues(predInput, runif(raster::ncell(predInput)))
  }, simplify=FALSE))

  names(predInput)<- names(df)

  res<- process(df, predInput=predInput, epochs=10, iterations=5, repVi=2, batch_size="all",
                DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)

  # TODO:
  # res<- process(df, predInput=predInput, responseVars=1:2, epochs=10, iterations=5, repVi=2, batch_size="all",
  #               DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)
})
