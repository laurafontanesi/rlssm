import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from .utils import hdi, bci

def plot_posterior(x,
                   ax=None,
                   gridsize=100,
                   clip=None,
                   show_intervals="HDI",
                   alpha_intervals=.05,
                   color='grey',
                   intervals_kws=None,
                   **kwargs):
    """Plots a univariate distribution with Bayesian intervals for inference.

    By default, only plots the kernel density estimation using scipy.stats.gaussian_kde.

    Bayesian instervals can be also shown as shaded areas,
    by changing show_intervals to either BCI or HDI.

    Parameters
    ----------

    x : array-like
        Usually samples from a posterior distribution.

    ax : matplotlib.axes.Axes, optional
        If provided, plot on this Axes.
        Default is set to current Axes.

    gridsize : int, default to 100
        Resolution of the kernel density estimation function.

    clip : tuple of (float, float), optional
        Range for the kernel density estimation function.
        Default is min and max values of `x`.

    show_intervals : str, default to "HDI"
        Either "HDI", "BCI", or None.
        HDI is better when the distribution is not simmetrical.
        If None, then no intervals are shown.

    alpha_intervals : float, default to .05
        Alpha level for the intervals calculation.
        Default is 5 percent which gives 95 percent BCIs and HDIs.

    intervals_kws : dict, optional
        Additional arguments for `matplotlib.axes.Axes.fill_between`
        that shows shaded intervals.
        By default, they are 50 percent transparent.

    color : matplotlib.colors
        Color for both the density curve and the intervals.

    Returns
    -------

    ax : matplotlib.axes.Axes
        Returns the `matplotlib.axes.Axes` object with the plot
        for further tweaking.

    """
    if clip is None:
        min_x = np.min(x)
        max_x = np.max(x)
    else:
        min_x, max_x = clip

    if ax is None:
        ax = plt.gca()

    if intervals_kws is None:
        intervals_kws = {'alpha':.5}

    density = gaussian_kde(x, bw_method='scott')
    xd = np.linspace(min_x, max_x, gridsize)
    yd = density(xd)

    ax.plot(xd, yd, color=color, **kwargs)

    if show_intervals is not None:
        if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
            raise ValueError("must be either None, BCI, or HDI")
        if show_intervals == 'BCI':
            low, high = bci(x, alpha_intervals)
        else:
            low, high = hdi(x, alpha_intervals)
        ax.fill_between(xd[np.logical_and(xd >= low, xd <= high)],
                        yd[np.logical_and(xd >= low, xd <= high)],
                        color=color,
                        **intervals_kws)
    return ax

def plot_mean_prediction(predictions,
                         data,
                         y_data,
                         y_predictions,
                         show_data=True,
                         color='grey',
                         ax=None,
                         **kwargs):
    if ax is None:
        ax = plt.gca()

    if show_data:
        data_mean = np.mean(data[y_data])
        ax.axvline(data_mean,
                   color=color)

    plot_posterior(predictions[y_predictions],
                   ax=ax,
                   color=color,
                   **kwargs)
    return ax

def plot_grouped_mean_prediction(x,
                                 y_data,
                                 y_predictions,
                                 predictions,
                                 data,
                                 hue=None,
                                 hue_order=None,
                                 x_order=None,
                                 hue_labels=None,
                                 show_data=True,
                                 show_intervals='HDI',
                                 alpha_intervals=.05,
                                 palette=None,
                                 color='grey',
                                 ax=None,
                                 intervals_kws=None):
    if ax is None:
        ax = plt.gca()

    if intervals_kws is None:
        intervals_kws = {'alpha':.5}

    if hue is None:
        if x_order is None:
            x_order = np.array(predictions.index.get_level_values(x).unique())

        if show_data:
            data_mean = [np.mean(data.loc[(data[x] == j).values, y_data]) for j in x_order]
            ax.plot(x_order,
                    data_mean,
                    color=color)

        if show_intervals is not None:
            if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
                raise ValueError("must be either None, BCI, or HDI")
            if show_intervals == 'BCI':
                low_high = [bci(predictions.loc[(j, slice(None)), y_predictions], alpha_intervals) for j in x_order]
                low = np.array(low_high)[:, 0]
                high = np.array(low_high)[:, 1]
            else:
                low_high = [hdi(predictions.loc[(j, slice(None)), y_predictions], alpha_intervals) for j in x_order]
                low = np.array(low_high)[:, 0]
                high = np.array(low_high)[:, 1]

            ax.fill_between(x_order,
                            low,
                            high,
                            low < high,
                            color=color,
                            **intervals_kws)
        ax.set_xlabel(x)
        ax.set_ylabel(y_predictions)
        #x_ = range(len(x_order))
        #plt.xticks(x_, x_order)
    else:
        if hue_order is None:
            hue_order = np.array(predictions.index.get_level_values(hue).unique())
        if x_order is None:
            x_order = np.array(predictions.index.get_level_values(x).unique())
        if hue_labels is None:
            hue_labels = hue_order
        if palette is None:
            palette = sns.husl_palette(n_colors=len(hue_labels))

        for i, cond in enumerate(hue_order):
            if show_data:
                data_mean = [np.mean(data.loc[np.logical_and(data[x] == j, data[hue] == cond), y_data]) for j in x_order]
                ax.plot(x_order,
                        data_mean,
                        color=palette[i],
                        label="Mean data (%s)" % hue_labels[i])


            if show_intervals is not None:
                if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
                    raise ValueError("must be either None, BCI, or HDI")
                if show_intervals == 'BCI':
                    low_high = [bci(predictions.loc[(j, cond, slice(None)), y_predictions], alpha_intervals) for j in x_order]
                    low = np.array(low_high)[:, 0]
                    high = np.array(low_high)[:, 1]
                else:
                    low_high = [hdi(predictions.loc[(j, cond, slice(None)), y_predictions], alpha_intervals) for j in x_order]
                    low = np.array(low_high)[:, 0]
                    high = np.array(low_high)[:, 1]

                ax.fill_between(x_order,
                                low,
                                high,
                                low < high,
                                color=palette[i],
                                label="{} prediction ({})".format(show_intervals, hue_labels[i]),
                                **intervals_kws)
        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_xlabel(x)
        ax.set_ylabel(y_predictions)
        #x_ = range(len(x_order))
        #plt.xticks(x_, x_order)
        return ax

def plot_quantiles_prediction(predictions,
                              data,
                              model,
                              quantiles=None,
                              show_data=True,
                              show_intervals='HDI',
                              kind='lines',
                              alpha_intervals=.05,
                              figsize=(15, 8),
                              color='grey',
                              scatter_kws=None,
                              intervals_kws=None):
    if quantiles is None:
        quantiles = [.1, .3, .5, .7, .9]

    percentiles = np.array(quantiles)*100

    if model == 'ddm':
        columns_up = ['quant_{}_rt_up'.format(int(p)) for p in percentiles]
        columns_low = ['quant_{}_rt_low'.format(int(p)) for p in percentiles]
    else:
        columns_up = ['quant_{}_rt_correct'.format(int(p)) for p in percentiles]
        columns_low = ['quant_{}_rt_incorrect'.format(int(p)) for p in percentiles]

    if scatter_kws is None:
        scatter_kws = {'marker':'x', 'lw':3, 's': 100}
    if intervals_kws is None:
        intervals_kws = {'alpha':.2}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Data
    if show_data:
        q_data_up = [np.nanpercentile(data.loc[data.accuracy == 1, 'rt'], q=x) for x in percentiles]
        q_data_low = [np.nanpercentile(data.loc[data.accuracy == 0, 'rt'], q=x) for x in percentiles]

        axes[0].scatter(quantiles, q_data_up, color=color, **scatter_kws)
        axes[1].scatter(quantiles, q_data_low, color=color, **scatter_kws)

    # Predictions
    if show_intervals is not None:
        if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
            raise ValueError("must be either None, BCI, or HDI")
        if show_intervals == 'BCI':
            q_pred_up = np.array([bci(predictions[x], alpha=alpha_intervals) for x in columns_up])
            q_pred_low = np.array([bci(predictions[x], alpha=alpha_intervals) for x in columns_low])
        else:
            q_pred_up = np.array([hdi(predictions[x], alpha=alpha_intervals) for x in columns_up])
            q_pred_low = np.array([hdi(predictions[x], alpha=alpha_intervals) for x in columns_low])

        if np.sum(kind == np.array(['lines', 'shades'])) < 1:
            raise ValueError("must be either lines or shades")
        if kind == 'lines':
            for i, q in enumerate(quantiles):
                axes[0].plot(
                    np.array([q, q]),
                    q_pred_up[i, :],
                    color=color,
                    **intervals_kws)
                axes[1].plot(
                    np.array([q, q]),
                    q_pred_low[i, :],
                    color=color,
                    **intervals_kws)

        else:
            axes[0].fill_between(
                quantiles,
                q_pred_up[:, 0],
                q_pred_up[:, 1],
                q_pred_up[:, 0] < q_pred_up[:, 1],
                color=color,
                **intervals_kws)
            axes[1].fill_between(
                quantiles,
                q_pred_low[:, 0],
                q_pred_low[:, 1],
                q_pred_low[:, 0] < q_pred_low[:, 1],
                color=color,
                **intervals_kws)
    for ax in axes:
        ax.set_xlabel('Quantiles')
        ax.set_xticks(quantiles)
        ax.set_xticklabels(quantiles)

    if model == 'ddm':
        axes[0].set_ylabel('RTs upper boundary')
        axes[1].set_ylabel('RTs lower boundary')
    else:
        axes[0].set_ylabel('RTs correct boundary')
        axes[1].set_ylabel('RTs incorrect boundary')

    sns.despine()
    return fig

def plot_grouped_quantiles_prediction(predictions,
                                      data,
                                      model,
                                      grouping_var,
                                      quantiles=None,
                                      show_data=True,
                                      show_intervals='HDI',
                                      kind='lines',
                                      alpha_intervals=.05,
                                      figsize=(20, 8),
                                      palette=None,
                                      hue_order=None,
                                      hue_labels=None,
                                      jitter=.05,
                                      scatter_kws=None,
                                      intervals_kws=None):

    if quantiles is None:
        quantiles = [.1, .3, .5, .7, .9]

    percentiles = np.array(quantiles)*100

    if model == 'ddm':
        columns_up = ['quant_{}_rt_up'.format(int(p)) for p in percentiles]
        columns_low = ['quant_{}_rt_low'.format(int(p)) for p in percentiles]
    else:
        columns_up = ['quant_{}_rt_correct'.format(int(p)) for p in percentiles]
        columns_low = ['quant_{}_rt_incorrect'.format(int(p)) for p in percentiles]


    if hue_order is None:
        hue_order = np.array(predictions.index.get_level_values(grouping_var).unique())
    if hue_labels is None:
        hue_labels = hue_order

    n_levels = len(hue_order)
    wide = jitter*(n_levels-1)
    jitter_levels = np.linspace(-wide/2, wide/2, n_levels)

    if palette is None:
        palette = sns.color_palette(n_colors=n_levels)
    if intervals_kws is None:
        intervals_kws = {'alpha':.2}
    if scatter_kws is None:
        scatter_kws = {'marker':'x', 'lw':3, 's': 100}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Data
    if show_data:
        for j, l in enumerate(hue_order):
            q_data_up = [np.nanpercentile(data.loc[np.logical_and(data[grouping_var] == l,
                                                                  data.accuracy == 1), 'rt'], q=x) for x in percentiles]
            q_data_low = [np.nanpercentile(data.loc[np.logical_and(data[grouping_var] == l,
                                                                   data.accuracy == 0), 'rt'], q=x) for x in percentiles]

            # Data
            axes[0].scatter(quantiles + jitter_levels[j],
                            q_data_up,
                            color=palette[j],
                            **scatter_kws)
            axes[1].scatter(quantiles + jitter_levels[j],
                            q_data_low,
                            color=palette[j],
                            label='%s, Data' % hue_labels[j],
                            **scatter_kws)

    # Predictions
    if show_intervals is not None:
        for j, l in enumerate(hue_order):
            prediction_label = '%s, Predictions' % hue_labels[j]

            if np.sum(show_intervals == np.array(['BCI', 'HDI'])) < 1:
                raise ValueError("must be either None, BCI, or HDI")
            if show_intervals == 'BCI':
                q_pred_up = np.array([bci(predictions.loc[(l, slice(None)), x], alpha=alpha_intervals) for x in columns_up])
                q_pred_low = np.array([bci(predictions.loc[(l, slice(None)), x], alpha=alpha_intervals) for x in columns_low])
            else:
                q_pred_up = np.array([hdi(predictions.loc[(l, slice(None)), x], alpha=alpha_intervals) for x in columns_up])
                q_pred_low = np.array([hdi(predictions.loc[(l, slice(None)), x], alpha=alpha_intervals) for x in columns_low])

            if np.sum(kind == np.array(['lines', 'shades'])) < 1:
                raise ValueError("must be either lines or shades")
            if kind == 'lines':
                for i, q in enumerate(quantiles):
                    axes[0].plot(
                        np.array([q, q]) + jitter_levels[j],
                        q_pred_up[i, :],
                        color=palette[j],
                        **intervals_kws)
                    axes[1].plot(
                        np.array([q, q]) + jitter_levels[j],
                        q_pred_low[i, :],
                        color=palette[j],
                        label=prediction_label,
                        **intervals_kws)
                    prediction_label = ''

            else:
                axes[0].fill_between(
                    quantiles + jitter_levels[j],
                    q_pred_up[:, 0],
                    q_pred_up[:, 1],
                    q_pred_up[:, 0] < q_pred_up[:, 1],
                    color=palette[j],
                    **intervals_kws)
                axes[1].fill_between(
                    quantiles + jitter_levels[j],
                    q_pred_low[:, 0],
                    q_pred_low[:, 1],
                    q_pred_low[:, 0] < q_pred_low[:, 1],
                    color=palette[j],
                    label=prediction_label,
                    **intervals_kws)
    for ax in axes:
        ax.set_xlabel('Quantiles')
        ax.set_xticks(quantiles)
        ax.set_xticklabels(quantiles)

    if model == 'ddm':
        axes[0].set_ylabel('RTs upper boundary')
        axes[1].set_ylabel('RTs lower boundary')
    else:
        axes[0].set_ylabel('RTs correct boundary')
        axes[1].set_ylabel('RTs incorrect boundary')

    axes[1].legend(frameon=True, bbox_to_anchor=(1, 1))

    sns.despine()
    return fig
