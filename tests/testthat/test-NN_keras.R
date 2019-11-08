context("NN_keras")

test_that("Process works", {
  df<- data.frame(replicate(10, runif(100)))

  res<- process(df, epochs=10, iterations=5, repVi=2, batch_size="all",  DALEXexplainer=TRUE, crossValRatio=0.8, NNmodel=FALSE, verbose=0)
})
