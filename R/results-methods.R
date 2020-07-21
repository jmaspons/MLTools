#' summary.process_NN
#'
#' @param object a process_NN object
#'
#' @return
#' @export
#'
#' @examples
# @importFrom coda mcmc mcmc.list
summary.process_NN<- function(object, ...){
  out<- list()

  ## performance
  perf.summary<- lapply(object$performance, function(x){
    x<- summary(coda::mcmc(x))
    data.frame(t(c(x$statistics[c("Mean", "SD", "Naive SE")], x$quantiles)), check.names=FALSE)
  })

  out$performance.summary<- do.call(rbind, perf.summary)


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
              y<- summary(coda::mcmc(na.omit(y)))
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
        prediction.summary<- summarize_pred.Raster(object$predictions)
      }
    } else {
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

  class(out)<- "summary.process_NN"

  return(out)
}


#' plotVI.process_NN
#'
#' @param res
#' @param vi
#' @param dispersion sd, se or ci. Metric of the dispersion bars
#'
#' @return
#' @export
#'
#' @examples
#' @import ggplot2
plotVI.process_NN<- function(res, vi=c("ratio", "diff", "raw"), dispersion=c("sd", "se", "ci")){
  vi<- match.arg(vi)
  dispersion<- match.arg(dispersion)
  vi.summary<- summary.process_NN(res)$vi
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
  ggplot(viD) +
    geom_bar( aes(x=stats::reorder(rownames(viD), -viD$Mean), y=viD$Mean),
              stat="identity", fill="skyblue", alpha=0.7) +
    geom_errorbar( aes(x=rownames(viD), ymin=mins, ymax=maxs),
                   width=0.4, colour="orange", alpha=0.9, size=1.3) +
    theme(axis.text.x=element_text(angle=90, hjust=1, vjust=0.5),
          axis.ticks.y=element_line(0.4, 0.5, 0.6))
}
