from bokeh import palettes
from bokeh.models import Range1d, LinearAxis, LegendItem, Legend
from bokeh.plotting import figure


def bokeh_multiline(df, col_secondary):
    col_primary = list(set(df.columns) - set(col_secondary))
    df_secondary, df_primary = df[col_secondary], df[col_primary]
    len_secondary, len_primary = len(df_secondary.columns), len(df_primary.columns)
    palette = palettes.Category10_10[:len(df.columns)]

    p = figure(x_axis_type="datetime", plot_width=800, plot_height=300)
    # p.extra_y_ranges = {"foo": Range1d(start=-3, end=3)}
    p.extra_y_ranges = {"foo": Range1d(start=-15, end=10)}
    # Adding the second axis to the plot.
    p.add_layout(LinearAxis(y_range_name="foo"), 'right')

    l1 = p.multi_line(xs=[df_secondary.index] * len_secondary,
                      ys=[df_secondary[col] for col in df_secondary.columns], line_width=2,
                      y_range_name="foo", line_color=palette[:len_secondary])

    l2 = p.multi_line(xs=[df_primary.index] * len_primary,
                      ys=[df_primary[col] for col in df_primary.columns], line_width=2,
                      line_color=palette[len_secondary:])

    items2 = [LegendItem(label=name, renderers=[l1], index=i)
              for i, name in enumerate(df_secondary.columns)]
    items1 = [LegendItem(label=name, renderers=[l2], index=i)
              for i, name in enumerate(df_primary.columns)]

    p.add_layout(Legend(items = items1+items2))

    return p