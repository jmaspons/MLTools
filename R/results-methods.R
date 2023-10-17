#' summary.pipe_result
#'
#' @param object a `pipe_result` object.
#' @param ... parameters to summarize_pred.Raster function.
#'
#' @return
#' @export
#'
#' @examples
# @importFrom coda mcmc mcmc.list
summary.pipe_result<- function(object, ...){
  out<- list()

  ## performance
  perf.summary<- lapply(object$performance, function(x){
    x<- summary(coda::mcmc(stats::na.omit(x)))
    data.frame(t(c(x$statistics[c("Mean", "SD", "Naive SE")], x$quantiles)), check.names=FALSE)
  })

  out$performance<- do.call(rbind, perf.summary)

  # SHAP
  if (!is.null(object$shap)){
    if (!inherits(object$shap, "shapviz")){ # aggregate_shap
      out$shap<- summary(do.call(rbind, object$shap))
    } else {
      out$shap<- summary(object$shap)
    }
  }

  ## variable importance
  if (!is.null(object$vi)){
    out$vi<- list()
    vi.summary<- apply(object$vi, 1, function(x){
      x<- summary(coda::mcmc(x))
      data.frame(t(c(x$statistics[c("Mean", "SD", "Naive SE")], x$quantiles)), check.names=FALSE)
    })

    out$vi$vi.summary<- do.call(rbind, vi.summary)

    viRatio<- as.matrix(object$vi)["_full_model_",] / as.matrix(object$vi)[which(rownames(object$vi) != "_full_model_"),]
    viRatio<- as.data.frame(t(viRatio))
    viRatio<- split(viRatio, sapply(strsplit(rownames(viRatio), "_"), function(y) y[1]))
    viRatio<- coda::mcmc.list(lapply(viRatio, coda::mcmc))
    viRatio.summary<- summary(viRatio)
    viRatio.summary<- data.frame((cbind(viRatio.summary$statistics[, c("Mean", "SD", "Naive SE")], viRatio.summary$quantiles)), check.names=FALSE)
    viRatio.summary<- viRatio.summary[order(viRatio.summary[, "Mean"], decreasing=FALSE), ]

    out$vi$viRatio.summary<- viRatio.summary

    viDiff<- as.matrix(object$vi)["_full_model_",] - as.matrix(object$vi)[which(rownames(object$vi) != "_full_model_"),]
    viDiff<- as.data.frame(t(viDiff))
    viDiff<- split(viDiff, sapply(strsplit(rownames(viDiff), "_"), function(y) y[1]))
    viDiff<- coda::mcmc.list(lapply(viDiff, coda::mcmc))
    viDiff.summary<- summary(viDiff)
    viDiff.summary<- data.frame((cbind(viDiff.summary$statistics[, c("Mean", "SD", "Naive SE")], viDiff.summary$quantiles)), check.names=FALSE)
    viDiff.summary<- viDiff.summary[order(viDiff.summary[, "Mean"], decreasing=FALSE), ]

    out$vi$viDiff.summary<- viDiff.summary
  }

  # variableResponse: omit
  # variableCoef
  if (!is.null(object$variableCoef)){
    out$variableCoef<- lapply(object$variableCoef, function(x){
        t(apply(x, 2, function(y){
              y<- summary(coda::mcmc(stats::na.omit(y)))
              c(y$statistics[c("Mean", "SD", "Naive SE")], y$quantiles)
        }))
    })
  }

  ## Predictions
  ## TODO: summarized.prediction in different raster brick for each response var
  if (!is.null(object$predictions)){
    if (inherits(object$predictions, "Raster") & requireNamespace("raster", quietly=TRUE)){
      if (all(names(object$predictions) == c("mean", "sd", "se"))){
        prediction.summary<- object$predictions
      } else {
        prediction.summary<- summarize_pred.Raster(pred=object$predictions, ...)
      }
    } else { # Non raster predictions
      prediction.summary<- lapply(object$predictions, function(x){
          if (all(names(x)[1:3] == c("Mean", "SD", "Naive SE"))){
            return(object$predictions)
          } else {
            return(summarize_pred(object$predictions))
          }
      })
    }

    out$prediction<- prediction.summary
  }

  out$params<- object$params

  class(out)<- paste0("summary.", class(object))

  return(out)
}


#' @export
print.pipe_result<- function(x, ...){
  if (inherits(x, "pipe_result.keras")){
    cat("Keras ")
  } else if (inherits(x, "pipe_result.randomForest")){
    cat("randomForest ")
  } else if (inherits(x, "pipe_result.xgboost")){
    cat("xgboost ")
  }
  cat("pipe result with", nrow(x$performance), "replicates.\n")

  cat("\nPerformance:\n")
  print(x$performance, ...)
  if (!is.null(x$shap)){
    cat("\nSHAP:\n")
    print(x$shap, ...)
  }
  if (!is.null(x$vi)){
    cat("\nVariable Importance:\n")
    print(x$vi, ...)
  }
  if (!is.null(x$variableCoef)){
    cat("\nlinear aproximations of the variables effects:\n")
    print(x$variableCoef, ...)
  }
  if (!is.null(x$predictions)){
    cat("\nPredictions for `predInput` data:\n")
    print(x$predictions, ...)
  }
  if (!is.null(x$model)){
    cat("Models saved in the results.\n")
  }
  if (!is.null(x$variableResponse)){
    cat("Variable response from ingredients::partial_dependency available in the results.\n")
  }
  if (!is.null(x$DALEXexplainer)){
    cat("DALEXexplainers saved in the results.\n")
  }
  if (!is.null(x$validateset)){
    cat("Validation set saved in the results.\n")
  }
  invisible(x)
}


#' @export
print.summary.pipe_result<- function(x, ...){
  if (inherits(x, "summary.pipe_result.keras")){
    cat("Keras ")
  } else if (inherits(x, "summary.pipe_result.randomForest")){
    cat("randomForest ")
  } else if (inherits(x, "summary.pipe_result.xgboost")){
    cat("xgboost ")
  }
  cat("pipe result summary with", nrow(x$performance), "replicates.\n")

  cat("\nPerformance:\n")
  print(x$performance, ...)
  if (!is.null(x$shap)){
    cat("\nSHAP:\n")
    print(x$shap, ...)
  }
  if (!is.null(x$vi)){
    cat("\nVariable Importance:\n")
    print(x$vi, ...)
  }
  if (!is.null(x$variableCoef)){
    cat("\nlinear aproximations of the variables effects:\n")
    print(x$variableCoef, ...)
  }
  if (!is.null(x$predictions)){
    cat("\nPredictions for `predInput` data:\n")
    print(x$predictions, ...)
  }
  if (!is.null(x$model)){
    cat("Models saved in the results.\n")
  }
  if (!is.null(x$variableResponse)){
    cat("Variable response from ingredients::partial_dependency available in the results.\n")
  }
  if (!is.null(x$DALEXexplainer)){
    cat("DALEXexplainers saved in the results.\n")
  }
  if (!is.null(x$validateset)){
    cat("Validation set saved in the results.\n")
  }
  invisible(x)
}


#' plotVI.pipe_result
#'
#' @param res a `pipe_result` object.
#' @param vi output type [`ratio`, `diff`, `raw`]. For ratio and diff, performance is compared with the full model without any permutation.
#' @param dispersion sd, se or ci. Metric of the dispersion bars
#'
#' @return
#' @export
#'
#' @examples
# @import ggplot2
plotVI.pipe_result<- function(res, vi=c("ratio", "diff", "raw"), dispersion=c("sd", "se", "ci")){
  vi<- match.arg(vi)
  dispersion<- match.arg(dispersion)
  vi.summary<- summary.pipe_result(res)$vi
  viD<- switch(vi,
             ratio=vi.summary$viRatio.summary,
             diff=vi.summary$viDiff.summary,
             raw=vi.summary$vi.summary)
  mins<- switch(dispersion,
            sd=viD$Mean - viD$SD,
            se=viD$Mean - viD$`Naive SE`,
            ci=viD$`2.5%`
          )
  maxs<- switch(dispersion,
            sd=viD$Mean + viD$SD,
            se=viD$Mean + viD$`Naive SE`,
            ci=viD$`97.5%`
          )
  ggplot2::ggplot(viD) +
    ggplot2::geom_bar( ggplot2::aes(x=stats::reorder(rownames(viD), -viD$Mean), y=viD$Mean),
              stat="identity", fill="skyblue", alpha=0.7) +
    ggplot2::geom_errorbar( ggplot2::aes(x=rownames(viD), ymin=mins, ymax=maxs),
                   width=0.4, colour="orange", alpha=0.9, size=1.3) +
    ggplot2::theme(axis.text.x=ggplot2::element_text(angle=90, hjust=1, vjust=0.5),
          axis.ticks.y=ggplot2::element_line(0.4, 0.5, 0.6))
}
