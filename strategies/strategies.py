import collections
import numpy as np
from scipy.stats import linregress
import utils.utils as utils
from statsmodels.tsa.arima.model import ARIMA as _ARIMA


class Strategy:
    def __init__(self, label="None", burn_in=0):
        self.label = label
        self.burn_in = burn_in
        Strategy.initialize(self)

    def initialize(self):
        self.returns = []
        self.dates = []
        self.ftpl_weight = 0.0
        self.yesterday_pos = 0
        self.loss = 0

    def update_returns(self, ret):
        value = utils.position_return(self.yesterday_pos, ret)
        self.returns.append(value)

    def update_dates(self, date):
        self.dates.append(date)

    def update_yesterday_pos(self, value):
        self.yesterday_pos = value

    def update_loss(self, position, ret):
        self.loss = utils.loss(position, ret)

    def update_weight(self, generator):
        self.ftpl_weight += self.loss + generator.exponential()


class MeanReversion(Strategy):
    """
    Simple mean reversion strategy that takes position that the difference between yesterday's
    return and the k-day average.
    """

    def __init__(self, k):
        """
        Initialize the class
        :param k: The mean reversion parameter (number of days to look back)
        """
        super().__init__(label=f"mr_{k}", burn_in=k)
        self.k = k
        self.initialize()

    def initialize(self):
        super().initialize()
        self.history = collections.deque(maxlen=self.k)

    def update(self, ret):
        """
        Calculate the position given the return
        :param ret: The return
        :return: The position (-1, 0, or 1)
        """
        pos = 0.0
        if len(self.history) == self.k:
            pos = -1.0 * np.sign(ret - np.mean(np.array(list(self.history))))
        self.history.append(ret)
        return pos


class Momentum(Strategy):
    """
    Simple momentum strategy that takes the return over the
    """

    def __init__(self, k):
        if k == 1:
            raise ValueError("window size k must be greater than 1.")
        super().__init__(label=f"momentum_{k}", burn_in=k)
        self.k = k
        self.initialize()

    def initialize(self):
        super().initialize()
        self.history = collections.deque(maxlen=self.k)

    def update(self, ret):
        """
        Calculate the position given the return
        :param ret: The return
        :return: The position (-1, 0, or 1)
        """
        pos = 0.0
        if len(self.history) == self.k:
            slope, intercept, r_val, p_val, std_err = linregress(
                range(self.k), self.history
            )
            pos = np.sign(slope)
        self.history.append(ret)
        return pos


class ARIMA(Strategy):
    def __init__(self, train_data, p, d, q):
        super().__init__(label=f"ARIMA_{p}_{d}_{q}", burn_in=max(p, d, q))
        self.train_data = train_data
        # self.p = p
        # self.d = d
        # self.q = q
        self.order = (p, d, q)
        self.initialize()

    def initialize(self):
        super().initialize()
        self.history = [x for x in self.train_data]

    def update(self, ret):
        """
        Calculate the position given the return
        :param ret: The return
        :return: The position (-1, 0, or 1)
        """
        self.history.append(ret)
        model = _ARIMA(self.history, order=self.order)
        model_fit = model.fit()
        output = model_fit.forecast()
        y_hat = output[0]
        pos = np.sign(y_hat - ret)
        return pos


class FTPL(Strategy):
    def __init__(self, seed=42):
        super().__init__(label="ftpl")
        self.gen = np.random.default_rng(seed=seed)
        self.initialize()

    def initialize(self):
        super().initialize()
        self.selection = []
        self.dates = []

    def update_selection(self, vector: np.array):
        self.selection.append(vector)

    def run_algorithm(self, returns: dict, models: list):
        # reset all models to maintain invariant
        self.initialize()
        [model.initialize() for model in models]
        counter = 0
        dates = sorted(list(returns))
        burn_in = 3 * max(model.burn_in for model in models)
        for num_day, date in enumerate(dates):
            counter += 1
            ret = returns[date]["return"]
            positions = [model.update(ret) for model in models]
            if counter > burn_in:
                date = returns[date]["date"]
                # calculate return for ftpl and for models
                self.update_dates(date)
                for i, model in enumerate(models):
                    # use the position taken the day prior
                    model.update_returns(ret)
                    # update the fpl algorithm
                    model.update_loss(positions[i], ret)
                    model.update_weight(self.gen)

                self.update_returns(ret)
                # gets the index of the model with max weight
                selected_model_idx = min(
                    range(len(models)), key=lambda i: models[i].ftpl_weight
                )
                selection_vector = np.array(
                    [int(i == selected_model_idx) for i in range(len(models))]
                )
                selected_position = positions[selected_model_idx]
                # update the selection array
                self.update_selection(selection_vector)
                # update the 'yesterday position' for ftpl model
                self.update_yesterday_pos(selected_position)
                # update the 'yesterday position' for all models
                for i, model in enumerate(models):
                    model.update_yesterday_pos(positions[i])
        self.selection = np.array(self.selection)
