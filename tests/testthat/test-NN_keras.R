context("NN_keras")

test_that("Trai", {
  df<- data.frame(replicate(10, runif(100)))

  res<- process(df, epochs=10, iterations=5, repVi=2, batch_size="all", verbose=0)
})
