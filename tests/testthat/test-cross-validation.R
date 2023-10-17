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

  # 2-Fold: test == validate (for faster testing)
  result$`2Fold`<- kFold_train_test_validate(d, k=2)
  result$`2Fold_weightClass`<- kFold_train_test_validate(d, k=2, replicates=replicates, caseClass=factor(caseClass), weight="class")
  # str(result)

  tmp<- lapply(result, function(x){
      lapply(x, lapply, function(y){
        expect_equal(names(y), intersect(names(y), c("trainset", "testset", "validateset", "weight.train", "weight.test", "weight.validate")))
      })
  })

})

