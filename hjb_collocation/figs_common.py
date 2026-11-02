import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "outputs", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 130, "font.size": 11, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
C_SPARSE, C_LEAST, C_INIT, C_OPT, C_TRUE = "#2563eb", "#dc2626", "#9ca3af", "#16a34a", "#111827"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)
