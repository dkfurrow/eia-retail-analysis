#!/usr/bin/env python
"""Reproduce WSJ graph with 0 y-axis
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as ml
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import datetime, date
from tabulate import tabulate
import os
import re
import itertools

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
# %%
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Year')
ax.set_ylabel('Price (\u00A2/kwh)')
ax.grid(True)
ax.set_ylim(0., 15.)
plot_schemes = [{'index_slice': ('DeReg', 'residential', 'wsjWtAvg'), 'color': 'orange', 'linestyle': '--', 'marker': 'o',
                 'label': """Retail Providers"""},
                {'index_slice': ('Reg', 'residential', 'wsjWtAvg'), 'color': 'grey', 'linestyle': ':', 'marker': 'o',
                 'label': """Traditional Utility"""}]
# Use linestyle keyword to style our plot
for plot_scheme in plot_schemes:
    ax.plot(aggregate_prices.columns.tolist(), aggregate_prices.loc[plot_scheme['index_slice'], :].values,
            color=plot_scheme['color'], linestyle=plot_scheme['linestyle'], marker=plot_scheme['marker'],
            label=plot_scheme['label'])
fig.canvas.draw()
ax.legend(loc='upper right')
plt.show()
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
