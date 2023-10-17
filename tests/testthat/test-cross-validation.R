context("Crossvalidation")

varScale<- seq(-100, 100, length.out=4)
names(varScale)<- paste0("X", 1:length(varScale))
d<- data.frame(lapply(varScale, function(i) runif(500) * i))
crossValRatio<- c(train=0.6, test=0.2, validate=0.2)
k<- 5
replicates<- 4
verbose<- 0
caseClass<- c(rep("A", 115), rep("B", 375), rep("C", 10))
weight<- "class"

test_that("bootstrap cross-validation works", {
  result<- list()

  result$bootstrap<- bootstrap_train_test_validate(d, replicates=replicates, ratio=crossValRatio)
  result$bootstrap_noValidation<- bootstrap_train_test_validate(d, replicates=replicates, ratio=c(train=0.8, test=.2, validate=0))
  result$bootstrap_noValidation<- bootstrap_train_test_validate(d, replicates=replicates, ratio=c(train=0.8, test=.2))
  result$bootstrap_noValidation<- bootstrap_train_test_validate(d, replicates=replicates, ratio=0.7)
  result$bootstrap_weightClass<- bootstrap_train_test_validate(d, replicates=replicates, ratio=crossValRatio, caseClass=caseClass, weight="class")
  result$bootstrap_weightClass_factor<- bootstrap_train_test_validate(d, replicates=replicates, ratio=crossValRatio, caseClass=factor(caseClass), weight="class")

  result$kFold<- kFold_train_test_validate(d, k=k)
  result$kFold_weightClass<- kFold_train_test_validate(d, k=k, replicates=replicates, caseClass=caseClass, weight="class")
  result$kFold_weightClass_factor<- kFold_train_test_validate(d, k=k, replicates=replicates, caseClass=factor(caseClass), weight="class")

  # 2-Fold: missing validate (lenght = 0)
  result$`2Fold`<- kFold_train_test_validate(d, k=2)
  result$`2Fold_weightClass`<- kFold_train_test_validate(d, k=2, replicates=replicates, caseClass=factor(caseClass), weight="class")
  # 1-Fold: test = train and missing validate
  result$`1Fold`<- kFold_train_test_validate(d, k=1)
  result$`1Fold_weightClass`<- kFold_train_test_validate(d, k=1, replicates=replicates, caseClass=factor(caseClass), weight="class")

  tmp<- lapply(result[grep("^2Fold", names(result))], function(x){
    sapply(x[grep("validate", names(x))], function(y) expect_length(y, 0))
    sapply(x[grep("validate", names(x), invert=TRUE)], function(y) expect_gt(length(y), 0))
  })

  tmp<- lapply(result[grep("^1Fold", names(result))], function(x){
    sapply(x[grep("validate", names(x))], function(y) expect_length(y, 0))
    sapply(x[grep("validate", names(x), invert=TRUE)], function(y) expect_gt(length(y), 0))
    mapply(function(train, test){
      expect_equal(train, test)
    }, train=x[grep("train", names(x))], test=x[grep("train", names(x))])
  })

  tmp<- lapply(result[grep("^[12]Fold", names(result), invert=TRUE)], function(x){
    sapply(x, function(y) expect_gt(length(y), 0))
    mapply(function(train, test, validate){
      expect_equal(lenght(c(train, test, train)), lenght(unique(c(train, test, train)))) # no duplicated idx among sets
    }, train=x[grep("train", names(x))], test=x[grep("train", names(x))], validate=x[grep("validate", names(x))])
  })

  tmp<- lapply(result, function(x){
      lapply(x, lapply, function(y){
        expect_equal(names(y), intersect(names(y), c("trainset", "testset", "validateset", "weight.train", "weight.test", "weight.validate")))
      })
  })
})

