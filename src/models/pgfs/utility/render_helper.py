import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from rdkit import Chem
from rdkit.Chem import QED, Draw


def render_steps(steps_log, save_path=None):
    # Pre-configure Matplotlib font settings
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    num_steps = len(steps_log)
    num_columns = num_steps + (num_steps + 1)

    fig = plt.figure(figsize=(num_columns * 2.5, 6), dpi=300)

    col_idx = 0
    for i in range(1, num_steps + 1):
        step_data = steps_log[i]

        # Initial reactant
        if i == 1:
            reactant_mol = Chem.MolFromSmiles(step_data["r1"])
            reactant_img = Draw.MolToImage(reactant_mol, size=(500, 500))
            ax = fig.add_axes([col_idx / num_columns, 0.3, 1 / num_columns, 0.5])
            ax.imshow(reactant_img)
            ax.axis("off")
            fig.text(
                col_idx / num_columns + 0.5 / num_columns,
                0.35,
                f"QED: {QED.qed(reactant_mol):.3f}",
                ha="center",
                fontsize=12,
            )
            col_idx += 1

        # Reaction arrow and template
        ax = fig.add_axes([(col_idx - 0.1) / num_columns, 0.3, 1 / num_columns, 0.5])
        arrow = FancyArrow(
            0.1,
            0.5,
            0.8,
            0,
            width=0.002,
            head_width=0.06,
            head_length=0.1,
            color="black",
        )
        ax.add_patch(arrow)
        ax.axis("off")
        fig.text(
            (col_idx - 0.1) / num_columns + 0.5 / num_columns,
            0.5,
            step_data["template"],
            ha="center",
            fontsize=10,
        )
        col_idx += 1

        # Second reactant (if available)
        if step_data["r2"]:
            second_reactant_mol = Chem.MolFromSmiles(step_data["r2"])
            second_reactant_img = Draw.MolToImage(second_reactant_mol, size=(500, 500))
            ax = fig.add_axes([(col_idx - 1) / num_columns, 0.6, 1 / num_columns, 0.4])
            ax.imshow(second_reactant_img)
            ax.axis("off")
            fig.text(
                (col_idx - 1.1) / num_columns + 0.5 / num_columns,
                0.6,
                "+",
                ha="center",
                fontsize=20,
            )

        # Product
        product_mol = Chem.MolFromSmiles(step_data["product"])
        product_img = Draw.MolToImage(product_mol, size=(500, 500))
        ax = fig.add_axes([col_idx / num_columns, 0.3, 1 / num_columns, 0.5])
        ax.imshow(product_img)
        ax.axis("off")
        fig.text(
            col_idx / num_columns + 0.5 / num_columns,
            0.35,
            f"QED: {QED.qed(product_mol):.3f}",
            ha="center",
            fontsize=12,
        )
        col_idx += 1

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
