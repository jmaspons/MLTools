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
      stop("`caseClass` parameter should be an atomic value indicating a column of d, a vector with values for every case in d or 1 if d is a vector containing caseClass.")
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

  out<- replicate(replicates, {
    subset_train_test_validate(d, ratio=ratio, caseClass=caseClass, weight=weight)
  }, simplify=FALSE)
  names(out)<- paste0("rep", formatC(1:length(out), format="d", flag="0", width=nchar(length(out))))

  return(out)
}


## K-fold cross validation ----
kFold_train_test_validate<- function(d, k=5, replicates=5, caseClass=NULL, weight="class"){
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
      stop("`caseClass` parameter should be an atomic value indicating a column of d, a vector with values for every case in d or 1 if d is a vector containing caseClass.")
    }
    indexL<- split(index, caseClass, drop=TRUE)
  }

  subsetIdxL<- lapply(indexL, function(idx){
    idx.idx<- caret::createMultiFolds(rep("", length(idx)), k=k, times=replicates)
    if (length(idx) < k){ # class with n < k
      nMiss<- k * replicates - length(idx.idx)
      if (nMiss > length(idx)){
        resample<- as.list(sample(length(idx), nMiss, replace=TRUE))
        warning("The number of samples in a case class category is << k. Sampling with replacement.")
      } else {
        resample<- as.list(sample(length(idx), nMiss))
      }
      idx.idx<- structure(c(idx.idx, resample), names=paste0("Fold", 1:k, ".Rep", rep(1:replicates, each=k)))
    }
    return(lapply(idx.idx, function(x) idx[x]))
  })


  ## Concatenate idx for each fold from different classes
  reps<- lapply(seq_along(subsetIdxL[[1]]), function(i){
    sort(structure(do.call(c, lapply(subsetIdxL, function(x) x[[i]])), names=NULL))
  })
  names(reps)<- names(subsetIdxL[[1]])

  repsL<- split(reps, gsub("^Fold[0-9]+\\.", "", names(reps)))

  if (k > 2){
    out<- lapply(repsL, function(x){
      idx.folds<- unique(unlist(x[-1], use.names=FALSE))
      validateset<- setdiff(idx.folds, x[[1]])
      lapply(x[-1], function(y){
        trainset<- setdiff(y, validateset)
        testset<- setdiff(idx.folds, c(trainset, validateset))
        list(trainset=trainset, testset=testset, validateset=validateset)
      })
    })
  } else if (k == 2){
    message("For k = 2, there are not enough folds for independent validatesets.")
    out<- lapply(repsL, function(x) {
      lapply(x[-1], function(y){
        trainset<- y
        testset<- x[[1]]
        list(trainset=trainset, testset=testset, validateset=integer())
      })
    })
  } else if (k == 1){
    message("For k = 1 doesn't make sense, there are not enough folds for independent test not validate sets. Will be the same as train set.")
    out<- lapply(repsL, function(x) {
      lapply(x, function(y){
        list(trainset=index, testset=index, validateset=integer())
      })
    })
  }

  out<- do.call(c, out)
  names(out)<- gsub("^Rep[0-9]+\\.", "", names(out))

  if (!is.null(caseClass)){
    if (all(weight == "class")){
      out<- lapply(out, function(x){
              w<- lapply(x, function(y){
                  weight.class(idx=y, caseClass=caseClass)
                })
              names(w)<- paste0("weight.", gsub("set$", "", names(w)))
              c(x, w)
            })
    }
    # CHECK: sapply(out, function(x) sapply(x[grep("weight", names(x))], sum))
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
