import numpy as np
from scipy.stats import norm
import pyperclip

def get_inv_norm_table(conf_row, conf_col):

    latex = "\\begin{table}[H]\n"
    latex += "    \\centering\n"
    latex += "    \\small\n"
    latex += "    \\begin{tabular}{|c|" + "c|" * len(conf_col) + "}\n"
    latex += "        \\hline\n"
    latex += "        \\textbf{Confi} & " + " & ".join([f"\\textbf{{{col:.3f}}}" for col in conf_col]) + " \\\\\n"
    latex += "        \\hline\n"

    for base in conf_row:
        line = f"        \\textbf{{{base:.2f}}}"
        for extra in conf_col:
            total_conf = base + extra
            if total_conf >= 1.0:
                line += " & ---"
            else:
                z = norm.ppf(total_conf)
                line += f" & {z:.5f}"
        latex += line + " \\\\\n"
        latex += "        \\hline\n"

    latex += "    \\end{tabular}\n"
    latex += "\\end{table}"
    return latex



conf_row1 = np.round(np.arange(0.80, 1, 0.01), 2)
conf_row2 = np.round(np.arange(0.90, 1.00, 0.01), 2)
conf_col = np.round(np.arange(0.00, 0.010, 0.001), 3)


res = get_inv_norm_table(conf_row1, conf_col)
pyperclip.copy(res)



