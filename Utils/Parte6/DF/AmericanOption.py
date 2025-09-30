import numpy as np

from Utils.Parte6.DF.EuropeanOption import CallEuropean, PutEuropean, BinaryCallEuropean, BinaryPutEuropean



class CallAmerican(CallEuropean):
    """
    American call option with optional discrete dividends.
    """

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


class PutAmerican(PutEuropean):
    """
    American put option with optional discrete dividends.
    """

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


class BinaryCallAmerican(BinaryCallEuropean):
    """
    American binary (cash-or-nothing) call option with optional discrete dividends.
    """

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


class BinaryPutAmerican(BinaryPutEuropean):
    """
    American binary (cash-or-nothing) put option with optional discrete dividends.
    """

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dividend_times = [0.25, 0.5, 0.75]
    dividend_amounts = [0, 0, 0]

    fig, ax = plt.subplots(figsize=(12, 10))
    option = CallAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Call American")
    option = CallEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Call European")

    fig, ax = plt.subplots(figsize=(12, 10))
    option = PutAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Put American")
    option = PutEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Put European")

    fig, ax = plt.subplots(figsize=(12, 10))
    option = BinaryCallAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call American")
    option = BinaryCallEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put European")

    fig, ax = plt.subplots(figsize=(12, 10))
    option = BinaryPutAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put American")
    option = BinaryPutEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put European")



    plt.show()