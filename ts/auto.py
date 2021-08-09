import numpy as np

from sklearn.linear_model import LinearRegression

import functools
import time
import warnings

from pmdarima.arima import _doc
from pmdarima.arima import _validation as val
from pmdarima.arima.utils import ndiffs, is_constant, nsdiffs
from pmdarima.utils import diff, is_iterable, check_endog
# Import as a namespace so we can mock
from pmdarima.arima import _auto_solvers as solvers
from pmdarima.compat.numpy import DTYPE
from pmdarima.compat import statsmodels as sm_compat
from pmdarima.compat import pmdarima as pm_compat
from pmdarima.arima.auto import _warn_for_deprecations

# __all__ = [
#     'auto_arima_lite'
# ]

def auto_arima_lite(y,
               X=None,
               start_p=2,
               d=None,
               start_q=2,
               max_p=5,
               max_d=2,
               max_q=5,
               start_P=1,
               D=None,
               start_Q=1,
               max_P=2,
               max_D=1,
               max_Q=2,
               max_order=5,
               m=1,
               seasonal=True,
               stationary=False,
               information_criterion='aic',
               alpha=0.05,
               test='kpss',
               seasonal_test='ocsb',
               stepwise=True,
               n_jobs=1,
               start_params=None,
               trend=None,
               method='lbfgs',
               maxiter=50,
               offset_test_args=None,
               seasonal_test_args=None,
               suppress_warnings=True,
               error_action='trace',
               trace=False,
               random=False,
               random_state=None,
               n_fits=10,
               return_valid_fits=False,
               out_of_sample_size=0,
               scoring='mse',
               scoring_args=None,
               with_intercept="auto",
               sarimax_kwargs=None,
               **fit_args):

    # NOTE: Doc is assigned BELOW this function

    # Temporary shim until we remove `exogenous` support completely
    X, fit_args = pm_compat.get_X(X, **fit_args)

    # pop out the deprecated kwargs
    fit_args = _warn_for_deprecations(**fit_args)

    # misc kwargs passed to various fit or test methods
    offset_test_args = val.check_kwargs(offset_test_args)
    seasonal_test_args = val.check_kwargs(seasonal_test_args)
    scoring_args = val.check_kwargs(scoring_args)
    sarimax_kwargs = val.check_kwargs(sarimax_kwargs)

    m = val.check_m(m, seasonal)
    trace = val.check_trace(trace)
    # can't have stepwise AND parallel
    n_jobs = val.check_n_jobs(stepwise, n_jobs)

    # validate start/max points
    start_p, max_p = val.check_start_max_values(start_p, max_p, "p")
    start_q, max_q = val.check_start_max_values(start_q, max_q, "q")
    start_P, max_P = val.check_start_max_values(start_P, max_P, "P")
    start_Q, max_Q = val.check_start_max_values(start_Q, max_Q, "Q")

    # validate d & D
    for _d, _max_d in ((d, max_d), (D, max_D)):
        if _max_d < 0:
            raise ValueError('max_d & max_D must be positive integers (>= 0)')
        if _d is not None:
            if _d < 0:
                raise ValueError('d & D must be None or a positive '
                                 'integer (>= 0)')

    # check on n_iter
    if random and n_fits < 0:
        raise ValueError('n_iter must be a positive integer '
                         'for a random search')

    # validate error action
    actions = {'warn', 'raise', 'ignore', 'trace', None}
    if error_action not in actions:
        raise ValueError('error_action must be one of %r, but got %r'
                         % (actions, error_action))

    # start the timer after the parameter validation
    start = time.time()

    # copy array
    y = check_endog(y, dtype=DTYPE)
    n_samples = y.shape[0]

    # the workhorse of the model fits
    fit_partial = functools.partial(
        solvers._fit_candidate_model,
        start_params=start_params,
        trend=trend,
        method=method,
        maxiter=maxiter,
        fit_params=fit_args,
        suppress_warnings=suppress_warnings,
        trace=trace,
        error_action=error_action,
        scoring=scoring,
        out_of_sample_size=out_of_sample_size,
        scoring_args=scoring_args,
        information_criterion=information_criterion,
    )

    # check for constant data
    if is_constant(y):
        warnings.warn('Input time-series is completely constant; '
                      'returning a (0, 0, 0) ARMA.')

        return _return_wrapper(
            solvers._sort_and_filter_fits(
                fit_partial(
                    y,
                    X=X,
                    order=(0, 0, 0),
                    seasonal_order=(0, 0, 0, 0),
                    with_intercept=val.auto_intercept(
                        with_intercept, False),  # False for the constant model
                    **sarimax_kwargs
                )
            ),
            return_valid_fits, start, trace, information_criterion)

    information_criterion = \
        val.check_information_criterion(information_criterion,
                                        out_of_sample_size)

    # the R code handles this, but I don't think statsmodels
    # will even fit a model this smallpmdarima..
    # if n_samples <= 3:
    #     if information_criterion != 'aic':
    #         warnings.warn('n_samples (%i) <= 3 '
    #                       'necessitates using AIC' % n_samples)
    #     information_criterion = 'aic'

    # adjust max p, q -- R code:
    # max.p <- min(max.p, floor(serieslength/3))
    # max.q <- min(max.q, floor(serieslength/3))
    max_p = int(min(max_p, np.floor(n_samples / 3)))
    max_q = int(min(max_q, np.floor(n_samples / 3)))

    # this is not in the R code and poses a risk that R did not considerpmdarima..
    # if max_p|q has now dropped below start_p|q, correct it.
    start_p = min(start_p, max_p)
    start_q = min(start_q, max_q)

    # if it's not seasonal, we can avoid multiple 'if not is None' comparisons
    # later by just using this shortcut (hack):
    # TODO: can we remove this hack now?
    if not seasonal:
        D = m = -1

    # TODO: check rank deficiency, check for constant Xs, regress if necessary
    xx = y.copy()
    if X is not None:
        lm = LinearRegression().fit(X, y)
        xx = y - lm.predict(X)

    # choose the order of differencing
    # is the TS stationary?
    if stationary:
        d = D = 0

    # todo: or not seasonal ?
    if m == 1:
        D = max_P = max_Q = 0
    # m must be > 1 for nsdiffs
    elif D is None:  # we don't have a D yet and we need one (seasonal)
        D = nsdiffs(xx, m=m, test=seasonal_test, max_D=max_D,
                    **seasonal_test_args)

        if D > 0 and X is not None:
            diffxreg = diff(X, differences=D, lag=m)
            # check for constance on any column
            if np.apply_along_axis(is_constant, arr=diffxreg, axis=0).any():
                D -= 1

    # D might still be None if not seasonal
    if D > 0:
        dx = diff(xx, differences=D, lag=m)
    else:
        dx = xx

    # If D was too big, we might have gotten rid of x altogether!
    if dx.shape[0] == 0:
        raise ValueError("The seasonal differencing order, D=%i, was too "
                         "large for your time series, and after differencing, "
                         "there are no samples remaining in your data. "
                         "Try a smaller value for D, or if you didn't set D "
                         "to begin with, try setting it explicitly. This can "
                         "also occur in seasonal settings when m is too large."
                         % D)

    # difference the exogenous matrix
    if X is not None:
        if D > 0:
            diffxreg = diff(X, differences=D, lag=m)
        else:
            diffxreg = X
    else:
        # here's the thingpmdarima.. we're only going to use diffxreg if exogenous
        # was not None in the first place. However, PyCharm doesn't know that
        # and it thinks we might use it before assigning it. Therefore, assign
        # it to None as a default value and it won't raise the warning anymore.
        diffxreg = None

    # determine/set the order of differencing by estimating the number of
    # orders it would take in order to make the TS stationary.
    if d is None:
        d = ndiffs(
            dx,
            test=test,
            alpha=alpha,
            max_d=max_d,
            **offset_test_args,
        )

        if d > 0 and X is not None:
            diffxreg = diff(diffxreg, differences=d, lag=1)

            # if any columns are constant, subtract one order of differencing
            if np.apply_along_axis(is_constant, arr=diffxreg, axis=0).any():
                d -= 1

    # check differences (do we want to warn?pmdarima..)
    if not suppress_warnings:  # TODO: context manager for entire block  # noqa: E501
        val.warn_for_D(d=d, D=D)

    if d > 0:
        dx = diff(dx, differences=d, lag=1)

    # check for constant
    if is_constant(dx):
        ssn = (0, 0, 0, 0) if not seasonal \
            else sm_compat.check_seasonal_order((0, D, 0, m))

        # Include the benign `ifs`, because R's auto.arima does. R has some
        # more options to control that we don't, but this is more readable
        # with a single `else` clause than a complex `elif`.
        if D > 0 and d == 0:
            with_intercept = val.auto_intercept(with_intercept, True)
            # TODO: if ever implemented in sm
            # fixed=mean(dx/m, na.rm = TRUE)
        elif D > 0 and d > 0:
            pass
        elif d == 2:
            pass
        elif d < 2:
            with_intercept = val.auto_intercept(with_intercept, True)
            # TODO: if ever implemented in sm
            # fixed=mean(dx, na.rm = TRUE)
        else:
            raise ValueError('data follow a simple polynomial and are not '
                             'suitable for ARIMA modeling')

        # perfect regression
        return _return_wrapper(
            solvers._sort_and_filter_fits(
                fit_partial(
                    y,
                    X=X,
                    order=(0, d, 0),
                    seasonal_order=ssn,
                    with_intercept=with_intercept,
                    **sarimax_kwargs
                )
            ),
            return_valid_fits, start, trace, information_criterion
        )

    # seasonality issues
    if m > 1:
        if max_P > 0:
            max_p = min(max_p, m - 1)
        if max_Q > 0:
            max_q = min(max_q, m - 1)

    # TODO: if approximation
    #   . we need method='css' or something similar for this

    # R determines whether to use a constant like this:
    #   allowdrift <- allowdrift & (d + D) == 1
    #   allowmean <- allowmean & (d + D) == 0
    #   constant <- allowdrift | allowmean
    # but we don't have `allowdrift` or `allowmean` so use just d and D
    if with_intercept == 'auto':
        with_intercept = (d + D) in (0, 1)

    if not stepwise:

        # validate max_order
        if max_order is None:
            max_order = np.inf
        elif max_order < 0:
            raise ValueError('max_order must be None or a positive '
                             'integer (>= 0)')

        search = solvers._RandomFitWrapper(
            y=y,
            X=X,
            fit_partial=fit_partial,
            d=d,
            D=D,
            m=m,
            max_order=max_order,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            random=random,
            random_state=random_state,
            n_fits=n_fits,
            n_jobs=n_jobs,
            seasonal=seasonal,
            trace=trace,
            with_intercept=with_intercept,
            sarimax_kwargs=sarimax_kwargs,
        )

    else:
        if n_samples < 10:
            start_p = min(start_p, 1)
            start_q = min(start_q, 1)
            start_P = start_Q = 0

        # seed p, q, P, Q vals
        p = min(start_p, max_p)
        q = min(start_q, max_q)
        P = min(start_P, max_P)
        Q = min(start_Q, max_Q)

        # init the stepwise model wrapper
        search = solvers._StepwiseFitWrapper(
            y,
            X=X,
            start_params=start_params,
            trend=trend,
            method=method,
            maxiter=maxiter,
            fit_params=fit_args,
            suppress_warnings=suppress_warnings,
            trace=trace,
            error_action=error_action,
            out_of_sample_size=out_of_sample_size,
            scoring=scoring,
            scoring_args=scoring_args,
            p=p,
            d=d,
            q=q,
            P=P,
            D=D,
            Q=Q,
            m=m,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            seasonal=seasonal,
            information_criterion=information_criterion,
            with_intercept=with_intercept,
            **sarimax_kwargs,
        )

    sorted_res = search.solve()
    return _return_wrapper(sorted_res, return_valid_fits, start, trace, information_criterion)


# Assign the doc to the auto_arima func
auto_arima_lite.__doc__ = _doc._AUTO_ARIMA_DOCSTR.format(
    y=_doc._Y_DOCSTR,
    X=_doc._EXOG_DOCSTR,
    fit_args=_doc._FIT_ARGS_DOCSTR,
    sarimax_kwargs=_doc._SARIMAX_ARGS_DOCSTR,
    return_valid_fits=_doc._VALID_FITS_DOCSTR
)


def _return_wrapper(fits, return_all, start, trace, information_criterion):
    """If the user wants to get all of the models back, this will
    return a list of the ARIMA models, otherwise it will just return
    the model. If this is called from the end of the function, ``fits``
    will already be a list.

    We *know* that if a function call makes it here, ``fits`` is NOT None
    or it would have thrown an exception in :func:`_post_ppc_arima`.

    Parameters
    ----------
    fits : iterable or ARIMA
        The ARIMA(s)

    return_all : bool
        Whether to return all.
    """
    # make sure it's an iterable
    if not is_iterable(fits):
        fits = [fits]

    # whether to print the final runtime
    if trace:
        print('Total fit time: %.3f seconds' % (time.time() - start))

    # which to return? if not all, then first index (assume sorted)
    if not return_all:
        return (
            fits[0].order,
            fits[0].seasonal_order,
            getattr(fits[0], information_criterion)(),
            fits[0].with_intercept )
    return [(fit.order,
             fit.seasonal_order,
             getattr(fit, information_criterion)(),
             fit.with_intercept) for fit in fits]

