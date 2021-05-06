context("NN_keras")

varScale<- seq(-100, 100, length.out=4)
names(varScale)<- paste0("X", 1:length(varScale))
df<- data.frame(lapply(varScale, function(i) runif(100) * i))
predInput<- data.frame(lapply(varScale, function(i) runif(50) * i))
responseVars<- 1
crossValRatio<- c(train=0.6, test=.2, validate=0.2)
idVars<- character()
ratio<- c(train=0.6, test=0.2, validate=0.2)
epochs<- 5
replicates<- 2
repVi<- 3
summarizePred<- TRUE
hidden_shape<- 3
batch_size<- "all"
scaleDataset<- FALSE
tempdirRaster<- tempdir()
filenameRasterPred<- paste0(tempdirRaster, "/testMap.grd")
baseFilenameRasterPred<- paste0(tempdirRaster, "/testMap")
baseFilenameNN<- paste0(tempdir(), "/testNN")
variableResponse<- TRUE
DALEXexplainer<- TRUE
crossValRatio<- 0.7
NNmodel<- TRUE
verbose<- 0
caseClass<- c(rep("A", 23), rep("B", 77))
weight<- "class"

test_that("bootstrap cross-validation works", {
  result<- list()

  # future::plan(future::transparent)
  future::plan(future::multisession)

  system.time(result$bootstrap<- bootstrap_train_test_validate(df, ratio=ratio))
  system.time(result$bootstrap_noValidation<- bootstrap_train_test_validate(df, ratio=c(train=0.8, test=.2, validate=0)))
  system.time(result$bootstrap_weightClass<- bootstrap_train_test_validate(df, ratio=ratio, caseClass=caseClass, weight="class"))

  system.time(result$bootstrap_rep<- replicate(replicates, bootstrap_train_test_validate(df, ratio=ratio), simplify=FALSE))
  system.time(result$bootstrap_weightClass_rep<- replicate(replicates, bootstrap_train_test_validate(df, ratio=ratio, caseClass=caseClass, weight="class"), simplify=FALSE))

  # str(result)

  tmp<- lapply(result, function(x){
    expect_type(x, type="list")
  })

  tmp<- lapply(result, function(x){
    if (inherits(x[[1]], "list")){
      lapply(x, function(y){
        expect_equal(names(y), c("trainset", "testset", "validateset", "weight.train", "weight.test", "weight.validate")[1:length(y)])
      })
    } else {
      expect_equal(names(x), c("trainset", "testset", "validateset", "weight.train", "weight.test", "weight.validate")[1:length(x)])
    }
  })

})

