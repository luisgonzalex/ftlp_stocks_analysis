import numpy as np
import matplotlib.pyplot as plt
import pandas
from datetime import datetime


def loss(pos, ret):
    """
    The loss between the position and the return (0 if same sign, 1 if opposite signs)
    :param pos: The position taken
    :param ret: The return of the asset
    :return: The loss (0 if same sign, 1 if opposite signs)
    """
    return 1 - ((np.sign(pos) * np.sign(ret)) + 1) / 2


def position_return(pos, ret):
    """
    The return of the position given the return.
    :param pos: The position taken
    :param ret: The return of the asset
    :return: The return on the position
    """
    return pos * ret


def _get_x_dates(dates):
    """
    Get the dates for x axis
    :param dates: List of all dates
    :return: the index and strings for x ticks
    """
    dates = np.array(sorted(dates))
    x_idx = np.linspace(0, len(dates) - 1, 10).astype(int)
    x_dates = dates[x_idx]
    return x_idx, x_dates


def visualize_returns(dates, return_list, name_list, strategy="", black=None):
    """
    Visualize the cumulative returns
    :param dates: The dates
    :param return_list: List of strategy returns
    :param name_list: List of the strategy names
    """
    x_idx, x_dates = _get_x_dates(dates)
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(name_list):
        if black is None or name != black:
            plt.plot(np.cumsum(return_list[i]), label=name)
        else:
            plt.plot(np.cumsum(return_list[i]), label=name, color="k")
    plt.xticks(x_idx, x_dates, rotation=-90)
    plt.legend()
    plt.title(f"Returns for {strategy} strategy")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.show()


def visualize_weights(dates, fpl_selection, name_list):
    x_idx, x_dates = _get_x_dates(dates)
    plt.figure(figsize=(10, 4))
    for i, name in enumerate(name_list):
        plt.plot(fpl_selection[:, i], label=name)
    plt.xticks(x_idx, x_dates, rotation=-90)
    plt.legend()
    plt.title("fpl selection")
    plt.show()


def load_data(filename: str):
    returns = {}  # Dictionary {date: return}
    with open(filename) as f:
        fl = True
        for line in f:
            if fl:
                fl = False  # ignore the header
            else:
                parsed = line.strip().split(",")
                returns[parsed[1]] = {
                    "return": float(parsed[2]),
                    "date": parsed[3],
                }  # {unix_ts: {date, return}
    return returns


def visualize(experiments: dict, strategy: str, dates: list):
    models, ftpl = (
        experiments[strategy]["models"],
        experiments[strategy]["ftpl"],
    )
    model_returns = [model.returns for model in models]
    labels = [model.label for model in models]
    # # add in ftpl
    model_returns.append(ftpl.returns)
    labels.append(ftpl.label)
    ftpl_selection = ftpl.selection
    visualize_returns(dates, model_returns, labels, strategy=strategy, black="ftpl")
    visualize_weights(dates, ftpl_selection, labels[:-1])


def clean_yahoo_data(filename):
    df = pandas.read_csv(filename)
    df["return"] = df["Close"] - df["Open"]
    df["Unix Timestamp"] = [
        datetime.strptime(elt, "%Y-%m-%d").timestamp() for elt in df["Date"]
    ]
    df.drop(df.columns[[i for i in range(1, 7)]], axis=1, inplace=True)
    cols = df.columns.tolist()
    cols = cols[::-1]
    df = df[cols]
    return df
