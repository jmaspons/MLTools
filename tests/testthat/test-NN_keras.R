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


test_that("Future plans work", {
  future::plan(future::multisession(workers=3))
  # options(future.globals.onReference = "error")
  system.time(resSessions<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                   DALEXexplainer=FALSE, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
# Error in keras::reset_states(modelNN) : attempt to apply non-function
# Don't import/export python objects to/from code inside future for PSOCK and multisession clusters
# https://cran.r-project.org/web/packages/future/vignettes/future-4-non-exportable-objects.html

  future::plan(future::transparent)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                   DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  future::plan(future::multicore)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                   DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))

  future::plan(future::sequential)
  system.time(res<- process_keras(df=df, predInput=predInput, responseVars=responseVars, epochs=epochs, replicates=replicates, repVi=repVi, batch_size=batch_size,
                                   DALEXexplainer=DALEXexplainer, crossValRatio=crossValRatio, NNmodel=NNmodel, verbose=verbose))
})

