# (c) Przemyslaw Biecek and Hubert Baniecki, GPL-3.0. From ingredients package fork https://github.com/jmaspons/ingredients/tree/myDev at ecaa5a9e2580570687a3cf59f94c82f4b0e601fa
# Waiting for https://github.com/ModelOriented/ingredients/pull/142 and https://github.com/ModelOriented/ingredients/pull/143

#' Feature Importance
#'
#' This function calculates permutation based feature importance.
#' For this reason it is also called the Variable Dropout Plot.
#'
#' Find more details in the \href{https://ema.drwhy.ai/featureImportance.html}{Feature Importance Chapter}.
#'
#' @param x an explainer created with function [DALEX::explain()], or a model to be explained.
#' @param data validation dataset, will be extracted from `x` if it's an explainer. Can be a list of arrays for multiinput models.
#' NOTE: It is safer when target variable is not present in the `data`.
#' @param predict_function predict function, will be extracted from `x` if it's an explainer.
#' @param y true labels for `data`, will be extracted from `x` if it's an explainer.
#' @param label name of the model. By default it's extracted from the `class` attribute of the model.
#' @param loss_function a function thet will be used to assess variable importance.
#' @param ... other parameters passed to `predict_function`.
#' @param type character, type of transformation that should be applied for dropout loss.
#' "raw" results raw drop losses, "ratio" returns `drop_loss / drop_loss_full_model`
#' while "difference" returns `drop_loss - drop_loss_full_model`.
#' @param N number of observations that should be sampled for calculation of variable importance.
#' If `NULL` then variable importance will be calculated on whole dataset (no sampling).
#' @param n_sample alias for `N` held for backwards compatibility. number of observations that should be sampled for calculation of variable importance.
#' @param B integer, number of permutation rounds to perform on each variable. By default it's `10`.
#' @param variables vector of variables or a list of vectors for multiinput models. If `NULL` then variable importance will be tested for each variable from the `data` separately. By default `NULL`
#' @param variable_groups list of variables names vectors or a list of vectors for multiinput models. This is for testing joint variable importance.
#' If `NULL` then variable importance will be tested separately for `variables`.
#' By default `NULL`. If specified then it will override `variables`,  `perm_dim` and `comb_dims`.
#' @param perm_dim the dimensions to perform the permutations when `data` is a 3d array (e.g. \[case, time, variable\]).
#' If `perm_dim = 2:3`, it calculates the importance for each variable in the 2nd and 3rd dimensions.
#' For multiinput models, a list of dimensions in the same order than in  `data`. If  `NULL`, the default, take all dimensions except the first one (i.e. rows) which correspond to cases.
#' @param comb_dims if `TRUE`, do the permutations for each combination of the levels of the variables from 2nd and 3rd dimensions for input data with 3 dimensions. By default, `FALSE`.
#'
#' @references Explanatory Model Analysis. Explore, Explain, and Examine Predictive Models. \url{https://ema.drwhy.ai/}
#'
#' @return an object of the class `feature_importance`
#' @importFrom methods hasArg
#'
#' @examples
#' library("DALEX")
#' library("ingredients")
#'
#' model_titanic_glm <- glm(survived ~ gender + age + fare,
#'                          data = titanic_imputed, family = "binomial")
#'
#' explain_titanic_glm <- explain(model_titanic_glm,
#'                                data = titanic_imputed[,-8],
#'                                y = titanic_imputed[,8])
#'
#' fi_glm <- feature_importance(explain_titanic_glm, B = 1)
#' plot(fi_glm)
#'
#' \dontrun{
#'
#' fi_glm_joint1 <- feature_importance(explain_titanic_glm,
#'                    variable_groups = list("demographics" = c("gender", "age"),
#'                    "ticket_type" = c("fare")),
#'                    label = "lm 2 groups")
#'
#' plot(fi_glm_joint1)
#'
#' fi_glm_joint2 <- feature_importance(explain_titanic_glm,
#'                    variable_groups = list("demographics" = c("gender", "age"),
#'                                           "wealth" = c("fare", "class"),
#'                                           "family" = c("sibsp", "parch"),
#'                                           "embarked" = "embarked"),
#'                    label = "lm 5 groups")
#'
#' plot(fi_glm_joint2, fi_glm_joint1)
#'
#' library("ranger")
#' model_titanic_rf <- ranger(survived ~., data = titanic_imputed, probability = TRUE)
#'
#' explain_titanic_rf <- explain(model_titanic_rf,
#'                               data = titanic_imputed[,-8],
#'                               y = titanic_imputed[,8],
#'                               label = "ranger forest",
#'                               verbose = FALSE)
#'
#' fi_rf <- feature_importance(explain_titanic_rf)
#' plot(fi_rf)
#'
#' fi_rf <- feature_importance(explain_titanic_rf, B = 6) # 6 replications
#' plot(fi_rf)
#'
#' fi_rf_group <- feature_importance(explain_titanic_rf,
#'                    variable_groups = list("demographics" = c("gender", "age"),
#'                    "wealth" = c("fare", "class"),
#'                    "family" = c("sibsp", "parch"),
#'                    "embarked" = "embarked"),
#'                    label = "rf 4 groups")
#'
#' plot(fi_rf_group, fi_rf)
#'
#' HR_rf_model <- ranger(status ~., data = HR, probability = TRUE)
#'
#' explainer_rf  <- explain(HR_rf_model, data = HR, y = HR$status,
#'                          model_info = list(type = 'multiclass'))
#'
#' fi_rf <- feature_importance(explainer_rf, type = "raw",
#'                             loss_function = DALEX::loss_cross_entropy)
#' head(fi_rf)
#' plot(fi_rf)
#'
#' HR_glm_model <- glm(status == "fired"~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR, y = as.numeric(HR$status == "fired"))
#' fi_glm <- feature_importance(explainer_glm, type = "raw",
#'                              loss_function = DALEX::loss_root_mean_square)
#' head(fi_glm)
#' plot(fi_glm)
#'
#' }
#' @export
#' @rdname feature_importance
feature_importance <- function(x, ...)
  UseMethod("feature_importance")

#' @export
#' @rdname feature_importance
feature_importance.explainer <- function(x,
                                         loss_function = DALEX::loss_root_mean_square,
                                         ...,
                                         type = c("raw", "ratio", "difference"),
                                         n_sample = NULL,
                                         B = 10,
                                         variables = NULL,
                                         variable_groups = NULL,
                                         N = n_sample,
                                         label = NULL) {
  if (is.null(x$data)) stop("The feature_importance() function requires explainers created with specified 'data' parameter.")
  if (is.null(x$y)) stop("The feature_importance() function requires explainers created with specified 'y' parameter.")

  # extracts model, data and predict function from the explainer
  model <- x$model
  data <- x$data
  predict_function <- x$predict_function
  if (is.null(label)) {
    label <- x$label
  }
  y <- x$y

  feature_importance.default(model,
                             data,
                             y,
                             predict_function = predict_function,
                             loss_function = loss_function,
                             label = label,
                             type = type,
                             N = N,
                             n_sample = n_sample,
                             B = B,
                             variables = variables,
                             variable_groups = variable_groups,
                             ...
  )
}

#' @export
#' @rdname feature_importance
feature_importance.default <- function(x,
                                       data,
                                       y,
                                       predict_function = predict,
                                       loss_function = DALEX::loss_root_mean_square,
                                       ...,
                                       label = class(x)[1],
                                       type = c("raw", "ratio", "difference"),
                                       n_sample = NULL,
                                       B = 10,
                                       variables = NULL,
                                       N = n_sample,
                                       variable_groups = NULL,
                                       perm_dim = NULL,
                                       comb_dims = FALSE) {
  # start: checks for arguments
##  if (is.null(N) & methods::hasArg("n_sample")) {
##    warning("n_sample is deprecated, please update ingredients and DALEX packages to use N instead")
##    N <- list(...)[["n_sample"]]
##  }

  if (inherits(data, "list")) {
    res <- feature_importance.multiinput(x = x,
                                         data = data,
                                         y = y,
                                         predict_function = predict_function,
                                         loss_function = loss_function,
                                         ...,
                                         label = label,
                                         type = type,
                                         B = B,
                                         variables = variables,
                                         N = N,
                                         variable_groups = variable_groups,
                                         perm_dim = perm_dim,
                                         comb_dims = comb_dims)
    return (res)
  }

  if (!is.null(variable_groups)) {
    if (!inherits(variable_groups, "list")) stop("variable_groups should be of class list")

    wrong_names <- !all(sapply(variable_groups, function(variable_set) {
      all(variable_set %in% colnames(data))
    }))

    if (wrong_names) stop("You have passed wrong variables names in variable_groups argument")
    if (!all(sapply(variable_groups, class) == "character")) stop("Elements of variable_groups argument should be of class character")
    if (is.null(names(variable_groups))) warning("You have passed an unnamed list. The names of variable groupings will be created from variables names.")
  }
  type <- match.arg(type)
  B <- max(1, round(B))

  # Adding names for variable_groups if not specified
  if (!is.null(variable_groups) && is.null(names(variable_groups))) {
    names(variable_groups) <- sapply(variable_groups, function(variable_set) {
      paste0(variable_set, collapse = "; ")
    })
  }

  # if `variable_groups` are not specified, then extract from `variables`
  if (is.null(variable_groups)) {
    # if `variables` are not specified, then extract from data
    if (is.null(variables)) {
      variables <- colnames(data)
      names(variables) <- colnames(data)
    }
  } else {
    variables <- variable_groups
  }

  # start: actual calculations
  # one permutation round: subsample data, permute variables and compute losses
  sampled_rows <- 1:nrow(data)
  loss_after_permutation <- function() {
    if (!is.null(N)) {
      if (N < nrow(data)) {
        # sample N points
        sampled_rows <- sample(1:nrow(data), N)
      }
    }
    sampled_data <- data[sampled_rows, , drop = FALSE]
    observed <- y[sampled_rows]
    # loss on the full model or when outcomes are permuted
    loss_full <- loss_function(observed, predict_function(x, sampled_data, ...))
    loss_baseline <- loss_function(sample(observed), predict_function(x, sampled_data, ...))
    # loss upon dropping a single variable (or a single group)
    loss_features <- sapply(variables, function(variables_set) {
      ndf <- sampled_data
      ndf[, variables_set] <- ndf[sample(1:nrow(ndf)), variables_set]
      predicted <- predict_function(x, ndf, ...)
      loss_function(observed, predicted)
    })
    c("_full_model_" = loss_full, loss_features, "_baseline_" = loss_baseline)
  }
  # permute B times, collect results into single matrix
  raw <- replicate(B, loss_after_permutation())

  # main result df with dropout_loss averages, with _full_model_ first and _baseline_ last
  res <- apply(raw, 1, mean)
  res_baseline <- res["_baseline_"]
  res_full <- res["_full_model_"]
  res <- sort(res[!names(res) %in% c("_full_model_", "_baseline_")])
  res <- data.frame(
    variable = c("_full_model_", names(res), "_baseline_"),
    permutation = 0,
    dropout_loss = c(res_full, res, res_baseline),
    label = label,
    row.names = NULL
  )
  if (type == "ratio") {
    res$dropout_loss = res$dropout_loss / res_full
  }
  if (type == "difference") {
    res$dropout_loss = res$dropout_loss - res_full
  }


  # record details of permutations
  attr(res, "B") <- B

  if (B > 1) {
    res_B <- data.frame(
      variable = rep(rownames(raw), ncol(raw)),
      permutation = rep(seq_len(B), each = nrow(raw)),
      dropout_loss = as.vector(raw),
      label = label
    )

    # here mean full model is used (full model for given permutation is an option)
    if (type == "ratio") {
      res_B$dropout_loss = res_B$dropout_loss / res_full
    }
    if (type == "difference") {
      res_B$dropout_loss = res_B$dropout_loss - res_full
    }

    res <- rbind(res, res_B)
  }

  class(res) <- c("feature_importance_explainer", "data.frame")

  if(!is.null(attr(loss_function, "loss_name"))) {
    attr(res, "loss_name") <- attr(loss_function, "loss_name")
  }
  res
}


feature_importance.multiinput <- function(x,
                                          data,
                                          y,
                                          predict_function = predict,
                                          loss_function = DALEX::loss_root_mean_square,
                                          ...,
                                          label = class(x)[1],
                                          type = c("raw", "ratio", "difference"),
                                          B = 10,
                                          variables = NULL,
                                          N = NULL,
                                          variable_groups = NULL,
                                          perm_dim = NULL,
                                          comb_dims = FALSE) {
  # start: checks for arguments
  ##  if (is.null(N) & methods::hasArg("n_sample")) {
  ##    warning("n_sample is deprecated, please update ingredients and DALEX packages to use N instead")
  ##    N <- list(...)[["n_sample"]]
  ##  }

  if (is.null(perm_dim) | !is.null(variable_groups)) {
    perm_dim <- lapply(data, function(d) stats::setNames(2:length(dim(d)), nm=names(dimnames(d))[-1])) # all dims except first (rows) which correspond to cases
  }

  # Variables for the dimensions to permute
  varsL <- mapply(function(d, dim) {
    dimnames(d)[dim]
  }, d=data, dim=perm_dim, SIMPLIFY=FALSE)

  if (!is.null(variable_groups)) {
    if (!inherits(variable_groups, "list") | !all(sapply(variable_groups, inherits, "list")))
        stop("variable_groups should be of class list contining lists for each data input")

    wrong_names <- !all(mapply(function(variable_set, vars) {
      all(unlist(variable_set) %in% unlist(vars))
    }, variable_set=variable_groups, vars=varsL[names(variable_groups)]))

    if (wrong_names) stop("You have passed wrong variables names in variable_groups argument")
    if (!all(unlist(sapply(variable_groups, sapply, sapply, class)) == "character"))
      stop("Elements of variable_groups argument should be of class character")
    if (any(sapply(sapply(variable_groups, names), is.null))) {
      warning("You have passed an unnamed list. The names of variable groupings will be created from variables names.")
      # Adding names for variable_groups if not specified
      variable_groups <- lapply(variable_groups, function(variable_sets_input) {
        if (is.null(names(variable_sets_input))) {
          group_names <- sapply(variable_sets_input, function(v) paste(paste(names(v), sapply(v, paste, collapse="; "), sep="."), collapse = " | "))
          names(variable_sets_input) <- group_names
        }
        variable_sets_input
      })
    }
  }
  type <- match.arg(type)
  B <- max(1, round(B))

  # if `variable_groups` are not specified, then extract from `variables`
  if (is.null(variable_groups)) {
    # if `variables` are not specified, then extract from data
    if (is.null(variables)) {
      variables <- lapply(varsL, function(vars) {
        if (comb_dims) {
          vars <- expand.grid(vars, stringsAsFactors=FALSE, KEEP.OUT.ATTRS=FALSE) # All combinations for all dimensions in a dataset
          rownames(vars) <- apply(vars, 1, function(v) paste(v, collapse="|"))
          vars <- split(vars, rownames(vars))
          vars <- lapply(vars, as.list)
        } else {
          vars <- mapply(function(dim_var, dim_names) {
            v <- lapply(dim_var, function(v) stats::setNames(list(v), dim_names))
            stats::setNames(v, nm = dim_var)
          }, dim_var=vars, dim_names=names(vars), SIMPLIFY=FALSE)
          vars <- do.call(c, vars)
        }
        vars
      })
    }
  } else {
    variables <- variable_groups
  }

  # start: actual calculations
  # one permutation round: subsample data, permute variables and compute losses
  n_cases <- unique(sapply(data, nrow))
  if (length(n_cases) > 1) {
    stop("Number of cases among inputs in data are different.")
  }
  sampled_rows <- 1:n_cases

  loss_after_permutation <- function() {
    if (!is.null(N)) {
      if (N < n_cases) {
        # sample N points
        sampled_rows <- sample(1:n_cases, N)
      }
    }
    sampled_data <- lapply(data, function(d) {
      if (length(dim(d)) == 2) {
        sampled_data <- d[sampled_rows, , drop = FALSE]
      } else if (length(dim(d)) == 3) {
        sampled_data <- d[sampled_rows, , , drop = FALSE]
      }
      sampled_data
    })
    observed <- y[sampled_rows]
    # loss on the full model or when outcomes are permuted
    loss_full <- loss_function(observed, predict_function(x, sampled_data, ...))
    loss_baseline <- loss_function(sample(observed), predict_function(x, sampled_data, ...))
    # loss upon dropping a single variable (or a single group)
    loss_featuresL <- mapply(function(d, vars, input_data) {
      loss_features <- sapply(vars, function(variables_set) {
        ndf <- d
        dim_perm <- names(dimnames(ndf)) %in% names(variables_set)
        dims <- list()
        for (i in 2:length(dim_perm)) {  # First dimension for cases
          if (dim_perm[i]) {
            dims[[i]] <- variables_set[[names(dimnames(ndf))[i]]]
          } else {
            dims[[i]] <- 1:dim(ndf)[i]
          }
        }
        names(dims) <- names(dimnames(ndf))

        if (length(dim_perm) == 2) {
          ndf[, dims[[2]]] <- ndf[sample(1:nrow(ndf)), dims[[2]]]
        } else if (length(dim_perm) == 3) {
          ndf[, dims[[2]], dims[[3]]] <- ndf[sample(1:nrow(ndf)), dims[[2]], dims[[3]]]
        } else {
          stop("Dimensions for this kind of data is not implemented but should be easy. Contact with the developers.")
        }
        sampled_data[[input_data]] <- ndf
        predicted <- predict_function(x, sampled_data, ...)
        loss_function(observed, predicted)
      })
    }, d=sampled_data, vars=variables, input_data=seq_along(sampled_data), SIMPLIFY=FALSE)

    unlist(c("_full_model_" = loss_full, loss_featuresL, "_baseline_" = loss_baseline))
  }
  # permute B times, collect results into single matrix
  raw <- replicate(B, loss_after_permutation())

  # main result df with dropout_loss averages, with _full_model_ first and _baseline_ last
  res <- apply(raw, 1, mean)
  res_baseline <- res["_baseline_"]
  res_full <- res["_full_model_"]
  res <- sort(res[!names(res) %in% c("_full_model_", "_baseline_")])
  res <- data.frame(
    variable = gsub(paste0("^(", paste(names(data), collapse="|"), ")\\."), "\\1: ", c("_full_model_", names(res), "_baseline_")),
    permutation = 0,
    dropout_loss = c(res_full, res, res_baseline),
    label = label,
    row.names = NULL
  )
  if (type == "ratio") {
    res$dropout_loss = res$dropout_loss / res_full
  }
  if (type == "difference") {
    res$dropout_loss = res$dropout_loss - res_full
  }


  # record details of permutations
  attr(res, "B") <- B

  if (B > 1) {
    res_B <- data.frame(
      variable = gsub(paste0("^(", paste(names(data), collapse="|"), ")\\."), "\\1: ", rep(rownames(raw), ncol(raw))),
      permutation = rep(seq_len(B), each = nrow(raw)),
      dropout_loss = as.vector(raw),
      label = label
    )

    # here mean full model is used (full model for given permutation is an option)
    if (type == "ratio") {
      res_B$dropout_loss = res_B$dropout_loss / res_full
    }
    if (type == "difference") {
      res_B$dropout_loss = res_B$dropout_loss - res_full
    }

    res <- rbind(res, res_B)
  }

  class(res) <- c("feature_importance_explainer", "data.frame")

  if(!is.null(attr(loss_function, "loss_name"))) {
    attr(res, "loss_name") <- attr(loss_function, "loss_name")
  }
  res
}
