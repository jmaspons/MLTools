context("NN_keras")

df<- data.frame(replicate(10, runif(100)))
predInput<- data.frame(replicate(10, runif(50)))
responseVars<- 1
idVars<- character()
epochs<- 10
replicates<- 5
repVi<- 2
batch_size<- "all"
DALEXexplainer<- TRUE
crossValRatio<- 0.8
NNmodel<- FALSE
verbose<- 0

test_that("process works", {
  system.time(res<- process(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  resB<- process(df=df, predInput=predInput, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
})

test_that("process_keras works", {
  # future::plan(future::multisession(workers=1))
  # future::plan(future::transparent)
  future::plan(future::sequential)
  system.time(res2<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  res2B<- process_keras(df=df, predInput=predInput, responseVars=1:2, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                 DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
})

test_that("Predict with raster", {
  predInputR<- raster::raster(nrows=15, ncols=15)
  predInputR<- raster::stack(replicate(10, {
    raster::setValues(predInputR, runif(raster::ncell(predInputR)))
  }, simplify=FALSE))

  names(predInputR)<- names(df)

  resR<- process(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)

  res2R<- process_keras(df, predInput=predInputR, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                        DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose)
  # TODO:
  # res<- process(df, predInput=predInputR, responseVars=1:2, epochs=10, replicates=5, repVi=2, batch_size="all",
  #               DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)
})

