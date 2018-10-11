import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as osp

import seaborn as sns


def convert_var_by_day(data, var, db_id, freq='10Min', day=0, show_gb=False, draw=False, aggr='max'):


    database_was_id = data[data['DB_ID'] == db_id]
    times = pd.DatetimeIndex(database_was_id['TIME'])
    database_was_id['HOUR'] = times.hour
    database_was_id['DAY'] = times.day

    days = database_was_id['DAY'].unique()

    if show_gb:
        database_was_id[var] = to_gb(database_was_id[var])

    # first day seems contains normal state
    was_day = database_was_id[(database_was_id['DAY'] == days[day])]
    var_data = was_day.groupby(pd.Grouper(key='TIME', freq=freq))[var].agg(aggr)

    x_ticks = np.linspace(0, len(var_data), len(var_data) + 1)
    f, ax = plt.subplots(figsize=(16, 4))
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    ax.set(xticks=x_ticks)
    sns.tsplot(var_data, ax=ax, condition=[var], legend=True)
    plt.savefig(osp.join(var + '_' + freq + '_' + aggr + '_%d.pdf' % day))
    plt.show()
    plt.close()
    return var_data
