## Function for cross-validation and data sampling
# Lever, J., Krzywinski, M., & Altman, N. (2016). Model selection and overfitting. Nature Methods, 13(9), 703-704. https://doi.org/10.1038/nmeth.3968


## Bootstrap ----
subset_train_test_validate<- function(d, ratio=c(train=0.6, test=0.2, validate=0.2), caseClass=NULL, weight="class"){
  if (inherits(d, c("data.frame", "matrix"))){
    index<- 1:nrow(d)
  } else {
    if (is.numeric(d) & length(d) == 1){ # atomic numeric
      index<- 1:d
    } else{ # vector
      index<- 1:length(d)
    }
  }


  if (is.null(caseClass)){
    indexL<- list(all=index)
  } else {
    if (length(caseClass) == 1 & inherits(d, c("data.frame", "matrix"))){
      classCol<- caseClass
      caseClass<- d[, classCol]
    } else if (length(caseClass) == 1 & is.atomic(d)){
        if (caseClass == 1){
          classCol<- NULL
          caseClass<- d
        }
    } else if (length(caseClass) == length(index)){
      classCol<- NULL
    } else {
      stop("«caseClass» parameter should be an atomic value indicating a column of d, a vector with values for every case in d or 1 if d is a vector containing caseClass.")
    }
    indexL<- split(index, caseClass, drop=TRUE)
  }

  if (length(ratio) == 1){
    names(ratio)<- NULL
    ratio<- c(train=ratio, test=1 - ratio) # Assuming ratio is the train proportion and no validation set
  }

  nSubsetL<- lapply(indexL, function(x){
    out<- c(train=round(length(x) * ratio[["train"]]),
            test=round(length(x) * ratio[["test"]]))
    out[["validate"]]<- length(x) - sum(out)
    if (sum(out) != length(x)) warning("sum of cases in subsets != total number of cases")

    out
  })

  subsetIdxL<- mapply(function(idx, nSubset){
    trainIdx<- sample(idx, size=nSubset[["train"]], replace=FALSE)
    testIdx<- sample(setdiff(idx, trainIdx), size=nSubset[["test"]], replace=FALSE)
    validateIdx<- setdiff(idx, c(trainIdx, testIdx))

    out<- list(train=trainIdx, test=testIdx, validate=validateIdx)
  }, idx=indexL, nSubset=nSubsetL, SIMPLIFY=FALSE)

  out<- list(trainset=structure(do.call(c, lapply(subsetIdxL, function(x) x$train)), names=NULL),
             testset=structure(do.call(c, lapply(subsetIdxL, function(x) x$test)), names=NULL),
             validateset=structure(do.call(c, lapply(subsetIdxL, function(x) x$validate)), names=NULL))

  if (!is.null(caseClass)){
    if (all(weight == "class")){
      out.weight<- lapply(out, function(x) weight.class(idx=x, caseClass=caseClass))
      names(out.weight)<- paste0("weight.", gsub("set$", "", names(out)))
    }

    out<- c(out, out.weight)
    # CHECK: sapply(out.weight, function(x) sum(x))
  }

  return(out)
}


bootstrap_train_test_validate<- function(d, replicates=10, ratio=c(train=0.6, test=0.2, validate=0.2), caseClass=NULL, weight="class"){
  if (length(ratio) == 1){
    names(ratio)<- NULL
    ratio<- c(train=ratio, test=1 - ratio) # Assuming ratio is the train proportion and no validation set
  }
  idxL0<- subset_train_test_validate(d, ratio=ratio, caseClass=caseClass, weight=weight)
  out<- list(validateset=idxL0$validateset)
  if (!is.null(idxL0$weight.validate)) out$weight.validate<- idxL0$weight.validate

  out$replicates<- list(list(trainset=idxL0$trainset, testset=idxL0$testset))
  if (!is.null(idxL0$weight.train)) out$replicates[[1]]<- c(out$replicates[[1]], list(weight.train=idxL0$weight.train, weight.test=idxL0$weight.test))

  if (replicates == 1) return(out)

  idx.boot<- sort(c(out$replicates[[1]]$trainset, out$replicates[[1]]$testset))
  if (inherits(d, c("data.frame", "matrix"))){
    d.boot<- d[idx.boot, ]
  } else {
    if (is.numeric(d) & length(d) == 1){ # atomic numeric
      d.boot<- length(idx.boot)
    } else{ # vector
      d.boot<- d[idx.boot]
    }
  }


  if (is.null(caseClass)){
    caseClass.boot<- NULL
  } else {
    if (length(caseClass) == 1 & inherits(d, c("data.frame", "matrix"))){
      classCol<- caseClass
      caseClass.boot<- d.boot[, classCol]
    } else if (length(caseClass) == 1 & is.atomic(d)){
      if (caseClass == 1){
        classCol<- NULL
        caseClass.boot<- d.boot
      }
    } else if (length(caseClass) == sum(sapply(list(out$validateset, out$replicates[[1]]$trainset, out$replicates[[1]]$testset), length))){
      classCol<- NULL
      caseClass.boot<- caseClass[idx.boot]
    } else {
      stop("«caseClass» parameter should be an atomic value indicating a column of d, a vector with values for every case in d or 1 if d is a vector containing caseClass.")
    }
  }

  reps<- replicate(replicates - 1, {
    subset_train_test_validate(d.boot, ratio=ratio[c("train", "test")] / sum(ratio[c("train", "test")]), caseClass=caseClass.boot, weight=weight)
  }, simplify=FALSE)
  reps<- lapply(reps, function(x) x[grep("validate", names(x), invert=TRUE)])
  out$replicates<- c(out$replicates, reps)

  return(out)
}


## K-fold cross validation ----
kFold_train_test_validate<- function(d, k=5, caseClass=NULL, weight="class"){
  if (inherits(d, c("data.frame", "matrix"))){
    index<- 1:nrow(d)
  } else {
    if (is.numeric(d) & length(d) == 1){ # atomic numeric
      index<- 1:d
    } else{ # vector
      index<- 1:length(d)
    }
  }


  if (is.null(caseClass)){
    indexL<- list(all=index)
  } else {
    if (length(caseClass) == 1 & inherits(d, c("data.frame", "matrix"))){
      classCol<- caseClass
      caseClass<- d[, classCol]
    } else if (length(caseClass) == 1 & is.atomic(d)){
      if (caseClass == 1){
        classCol<- NULL
        caseClass<- d
      }
    } else if (length(caseClass) == length(index)){
      classCol<- NULL
    } else {
      stop("«caseClass» parameter should be an atomic value indicating a column of d, a vector with values for every case in d or 1 if d is a vector containing caseClass.")
    }
    indexL<- split(index, caseClass, drop=TRUE)
  }

  subsetIdxL<- lapply(indexL, function(idx){
    idx.idx<- caret::createFolds(rep("", length(idx)), k=k)
    if (length(idx.idx) < k){ # class with n < k
      nMiss<- k - length(idx.idx)
      if (nMiss > length(idx.idx)){
        resample<- sample(idx.idx, nMiss, replace=TRUE)
        warning("The number of samples in a case class category is << k. Sampling with replacement.")
      } else {
        resample<- sample(idx.idx, nMiss)
      }
      idx.idx<- structure(c(idx.idx, resample), names=paste0("Fold", 1:k))
    }
    lapply(idx.idx, function(x) idx[x])
  })


  reps<- lapply(1:k, function(i){
    sort(structure(do.call(c, lapply(subsetIdxL, function(x) x[[i]])), names=NULL))
  })


  idx.folds<- unlist(lapply(subsetIdxL, function(x) x[-1])) # reps[1] = validateset
  reps<- lapply(reps, function(x){
    testset<- x
    trainset<- setdiff(idx.folds, testset)

    return(list(trainset=trainset, testset=testset))
  })

  out<- list(validateset=reps[[1]]$testset,
             weight.validate=NULL, replicates=reps[-1])

  if (!is.null(caseClass)){
    if (all(weight == "class")){
      out$weight.validate<- weight.class(idx=out$validateset, caseClass=caseClass)

      out$replicates<- lapply(out$replicates, function(x){
          w<- lapply(x, function(y){
              weight.class(idx=y, caseClass=caseClass)
            })
          names(w)<- paste0("weight.", gsub("set$", "", names(w)))
          c(x, w)
        })
    }
    # CHECK: sum(out$weight.validate);  sapply(out$replicates, function(x) sapply(x[grep("weight", names(x))], sum))
  } else {
    out$weight.validate<- NULL
  }

  return(out)
}


## Weight samples ----
weight.class<- function(idx, caseClass, weight="class"){
  caseClass<- as.character(caseClass)
  caseClassIdx<- caseClass[idx]
  nClasses<- length(unique(caseClassIdx))

  nCases<- length(idx)
  weight<- sapply(unique(caseClassIdx), function(x){
    1 / nClasses / sum(caseClassIdx %in% x)
  })

  out<- numeric(nClasses)
  for (i in seq_along(weight)){
    out[caseClassIdx %in% names(weight)[i]]<- weight[names(weight)[i]]
  }

  return(out)
}
