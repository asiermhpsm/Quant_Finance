import numpy as np

from Utils.Parte6.DF.EuropeanOption import CallEuropean, PutEuropean, BinaryCallEuropean, BinaryPutEuropean



class CallAmerican(CallEuropean):
    """
    American call option with optional discrete dividends.
    """

    name = "American Call Option"

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


class PutAmerican(PutEuropean):
    """
    American put option with optional discrete dividends.
    """

    name = "American Put Option"

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


class BinaryCallAmerican(BinaryCallEuropean):
    """
    American binary (cash-or-nothing) call option with optional discrete dividends.
    """

    name = "American Binary Call Option"

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


class BinaryPutAmerican(BinaryPutEuropean):
    """
    American binary (cash-or-nothing) put option with optional discrete dividends.
    """

    name = "American Binary Put Option"

    def solve(self):
        super().solve()
        self.V = np.maximum(self.V, self.payoff(self.S))


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dividend_times = [0.25, 0.5, 0.75]
    dividend_amounts = [1, 5, 10]

    print("Testing Call Options with Dividends")
    fig, ax = plt.subplots(figsize=(12, 10))
    print("Call American")
    option = CallAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Call American")
    print("Call European")
    option = CallEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Call European")

    print("Testing Put Options with Dividends")
    fig, ax = plt.subplots(figsize=(12, 10))
    print("Put American")
    option = PutAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Put American")
    print("Put European")
    option = PutEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Put European")

    print("Testing Binary Call Options with Dividends")
    fig, ax = plt.subplots(figsize=(12, 10))
    print("Binary Call American")
    option = BinaryCallAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call American")
    print("Binary Call European")
    option = BinaryCallEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call European")

    print("Testing Binary Put Options with Dividends")
    fig, ax = plt.subplots(figsize=(12, 10))
    print("Binary Put American")
    option = BinaryPutAmerican(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put American")
    print("Binary Put European")
    option = BinaryPutEuropean(dividend_times=dividend_times, dividend_amounts=dividend_amounts)
    option.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put European")



    plt.show()