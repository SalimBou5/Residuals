from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / "myoleg_bilateral_mep_80.csv")
phase = df["gait_cycle_pct"].to_numpy()

right_muscles = sorted(c for c in df.columns if c.endswith("_r"))

NROWS, NCOLS = 5, 4
PER_PAGE = NROWS * NCOLS

muscles = [c for c in df.columns if c.endswith(("_r", "_l"))]
ymax = max(0.1, df[muscles].max().max() * 1.1)


def plot_mep(mode, pdf):
    """
    mode: 'right', 'left', or 'bilateral'
    """

    names = (
        right_muscles if mode != "left"
        else [m.replace("_r", "_l") for m in right_muscles]
    )

    total_pages = int(np.ceil(len(names) / PER_PAGE))

    for start in range(0, len(names), PER_PAGE):

        subset = names[start:start + PER_PAGE]
        page = start // PER_PAGE + 1

        fig, axes = plt.subplots(
            NROWS, NCOLS,
            figsize=(16, 12),
            sharex=True,
            sharey=True
        )
        axes = axes.flatten()

        for ax, muscle in zip(axes, subset):

            base = muscle[:-2]

            # Select which signals to display
            if mode == "bilateral":
                signals = [
                    (base + "_r", "tab:blue", "-", "Right"),
                    (base + "_l", "tab:orange", "-", "Left")
                ]
            else:
                signals = [
                    (muscle,
                     "tab:blue" if mode == "right" else "tab:orange",
                     "-", mode.capitalize())
                ]

            # Plot selected signals
            for name, color, style, label in signals:
                ax.plot(
                    phase,
                    df[name],
                    color=color,
                    linestyle=style,
                    linewidth=1.8,
                    label=label
                )

            ax.set_title(base, fontsize=11, fontweight="bold")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(labelbottom=True, labelsize=9)
            ax.grid(True, alpha=0.25)

            # Missing reference
            if all(np.allclose(df[name], 0) for name, *_ in signals):
                ax.text(
                    0.5, 0.5, "No reference signal",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, color="gray"
                )

        for ax in axes[len(subset):]:
            ax.set_visible(False)

        fig.suptitle(
            f"MyoLeg — {mode.capitalize()} MEP profiles "
            f"({page}/{total_pages})",
            fontsize=17, fontweight="bold"
        )

        if mode == "bilateral":
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.96),
                ncol=2, frameon=False
            )

        fig.supxlabel("Gait cycle (%)")
        fig.supylabel("MEP amplitude")
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.92])

        fig.savefig(
            HERE / f"outputs/plots/png/GV_MEP_all_muscles_myoleg_{mode}_{page}.png",
            dpi=250, bbox_inches="tight"
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.show()
        plt.close(fig)


# Original right + left plots (4 pages)
with PdfPages(HERE / "outputs/plots/GV_MEP_all_muscles_myoleg_bilateral.pdf") as pdf:
    plot_mep("bilateral", pdf)
    plot_mep("right", pdf)
    plot_mep("left", pdf)