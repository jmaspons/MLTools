context("NN_keras")

varScale<- seq(-100, 100, length.out=4)
names(varScale)<- paste0("X", 1:length(varScale))
df<- data.frame(lapply(varScale, function(i) runif(100) * i))
predInput<- data.frame(lapply(varScale, function(i) runif(50) * i))
responseVars<- 1
crossValRatio<- c(train=0.6, test=.2, validate=0.2)
idVars<- character()
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
k<- 5
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
NNmodel<- TRUE
verbose<- 0
caseClass<- c(rep("A", 23), rep("B", 75), rep("C", 2))
weight<- "class"

test_that("bootstrap cross-validation works", {
  result<- list()

  # future::plan(future::transparent)
  # future::plan(future::multisession)

  system.time(result$bootstrap<- bootstrap_train_test_validate(df, ratio=crossValRatio))
  system.time(result$bootstrap_noValidation<- bootstrap_train_test_validate(df, ratio=c(train=0.8, test=.2, validate=0)))
  system.time(result$bootstrap_noValidation<- bootstrap_train_test_validate(df, ratio=c(train=0.8, test=.2)))
  system.time(result$bootstrap_noValidation<- bootstrap_train_test_validate(df, ratio=0.7))
  system.time(result$bootstrap_weightClass<- bootstrap_train_test_validate(df, ratio=crossValRatio, caseClass=caseClass, weight="class"))
  system.time(result$bootstrap_weightClass<- bootstrap_train_test_validate(df, ratio=crossValRatio, caseClass=factor(caseClass), weight="class"))

  system.time(result$kFold<- kFold_train_test_validate(df, k=k))
  system.time(result$kFold_weightClass<- kFold_train_test_validate(df, k=k, caseClass=caseClass, weight="class"))
  system.time(result$kFold_weightClass<- kFold_train_test_validate(df, k=k, caseClass=factor(caseClass), weight="class"))

  # str(result)

  tmp<- lapply(result, function(x){
    expect_type(x, type="list")
    if (length(x) == 3){
      expect_equal(names(x), c("validateset", "weight.validate", "replicates"))
    } else {
      expect_equal(names(x), c("validateset", "replicates"))
    }
  })

  tmp<- lapply(result, function(x){
      lapply(x$replicates, function(y){
        expect_equal(names(y), c("trainset", "testset", "weight.train", "weight.test")[1:length(y)])
      })
  })

})

