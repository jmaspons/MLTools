## Function for cross-validation and data sampling

# Bootstrap
splitdf<- function(df, ratio=0.8, trainLimits=FALSE, sample_weight=NULL, seed) {
  if (!missing(seed)) set.seed(seed)
  index<- 1:nrow(df)

  if (trainLimits){
    limitindex<- apply(df[, apply(df, 2, is.numeric)], 2, function(x){
      # limit<- range(x)
      mins<- which(x %in% min(x))
      maxs<- which(x %in% max(x))
      c(sample(mins, 1), sample(maxs, 1))
    })
    limitindex<- unique(as.vector(unlist(limitindex)))
  } else {
    limitindex<- integer()
  }

  nTrain<- round(length(index) * ratio)

  if (length(limitindex) > nTrain){
    warning("The number of extrem cases is bigger than the number of cases used to train. Increase the train ratio or look for more data.")
    limitindex<- sample(limitindex, size=nTrain)
  }

  trainindex<- sample(setdiff(index, limitindex), size=nTrain - length(limitindex), replace=FALSE)
  trainindex<- c(limitindex, trainindex)
  trainset<- df[trainindex, ]
  testset<- df[-trainindex, ]

  out<- list(trainset=trainset, testset=testset)

  if (!is.null(sample_weight)){
    out$sample_weight.trainset<- sample_weight[trainindex]
    out$sample_weight.testset<- sample_weight[-trainindex]
  }

  return(out)
}

# Bootstrap
# Lever, J., Krzywinski, M., & Altman, N. (2016). Model selection and overfitting. Nature Methods, 13(9), 703-704. https://doi.org/10.1038/nmeth.3968
bootstrap_train_test_validate<- function(d, ratio=c(train=0.6, test=0.2, validate=0.2), caseClass=NULL, weight="class"){
  if (inherits(d, c("data.frame", "matrix"))){
    index<- 1:nrow(d)
  } else {
    if (is.numeric(d) & length(d) == 1){
      index<- 1:d
    } else{
      index<- 1:length(d)
    }
  }


  if (is.null(caseClass)){
    indexL<- list(all=index)
  } else {
    if (length(caseClass) == 1 & inherits(d, c("data.frame", "matrix"))){
      classCol<- caseClass
      caseClass<- d[, classCol]
    } else if (caseClass == 1 & is.atomic(d)){
      classCol<- NULL
      caseClass<- d
    } else if (length(caseClass) == length(index)){
      classCol<- NULL
    } else {
      stop("«caseClass» parameter should be an atomic value indicating a column of d, a vector with values for every case in d or 1 if d is a vector containing caseClass.")
    }
    indexL<- split(index, caseClass)
  }

  nSubsetL<- lapply(indexL, function(x){
    out<- c(train=round(length(x) * ratio[["train"]]),
            test=round(length(x) * ratio[["test"]]))
    out[["validate"]]<- length(x) - sum(out)
    if (sum(out) != length(x)) warning("sum of cases in subsets != total number of cases")

    out
  })

  ## TODO: take validate set and then replicates for test and train sets

  subsetIdxL<- mapply(function(idx, nSubset){
    trainIdx<- sample(idx, size=nSubset[["train"]], replace=FALSE)
    testIdx<- sample(setdiff(idx, trainIdx), size=nSubset[["test"]], replace=FALSE)
    validateIdx<- setdiff(idx, c(trainIdx, testIdx))

    out<- list(train=trainIdx, test=testIdx, validate=validateIdx)
  }, idx=indexL, nSubset=nSubsetL, SIMPLIFY=FALSE)

  out<- list(trainset=do.call(c, lapply(subsetIdxL, function(x) x$train)),
             testset=do.call(c, lapply(subsetIdxL, function(x) x$test)),
             validateset=do.call(c, lapply(subsetIdxL, function(x) x$validate)))
## TODO: factor out in a weight function
  out.weight<- NULL
  if (!is.null(weight)){
    if (all(weight == "class") & !is.null(caseClass)){
      # Weight cases to balance differences in number of cases among classes
      # propClass<- do.call(rbind, nSubsetL)
      # propClass<- t(propClass) / colSums(propClass)

      # TEST: d=subsetIdxL[[1]]; className=names(subsetIdxL)[1]; set="train"
      weightL<- mapply(function(idx, className){
        out<- lapply(names(idx), function(set){
          weightCase<- 1 / length(subsetIdxL) / length(idx[[set]])
          rep(weightCase, length(idx[[set]]))
        })
        names(out)<- names(idx)
        out
      }, idx=subsetIdxL, className=names(subsetIdxL), SIMPLIFY=FALSE)

      out.weight<- list(weight.train=do.call(c, lapply(weightL, function(x) x$train)),
                        weight.test=do.call(c, lapply(weightL, function(x) x$test)),
                        weight.validate=do.call(c, lapply(weightL, function(x) x$validate)))
    }

  }

  out<- c(out, out.weight)
# CHECK: sapply(out.weight, function(x) sum(x))
  return(out)
}

