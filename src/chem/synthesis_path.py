import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrow
from rdkit import Chem
from rdkit.Chem import Draw

from src.chem.chem_utils import get_compound_name


def print_path(df, path_id):
    # Filter the dataframe by the specified path_id
    path_data = df.loc[path_id]

    # Check if path_data is not empty
    if path_data.empty:
        print(f"No synthesis path found for path_id: {path_id}")
        return

    # Display the synthesis path
    print(f"Synthesis Path ID: {path_id}")
    for _, row in path_data.iterrows():
        print(f"Step {row['step']}:")
        print(
            f"  Reactant: {row['reactant']} | Name: {get_compound_name(row['reactant'])}"
        )
        if pd.notna(row["second_reactant"]):
            print(
                f"  Second Reactant: {row['second_reactant']} | Name: {get_compound_name(row['second_reactant'])}"
            )
        print(f"  Template: {row['template']}")
        print(
            f"  Product: {row['product']} | Name: {get_compound_name(row['product'])}"
        )
        print(f"  QED Score: {row['qed']}")
        print("-" * 40)


def draw_path(df, path_id):
    # Check if path_id exists in the DataFrame index
    if path_id not in df.index:
        return f"Error: path_id {path_id} does not exist in the DataFrame."

    # Filter the DataFrame for the chosen path_id
    path_df = df.loc[path_id]

    # Calculate number of columns needed for subplots
    num_steps = len(path_df)
    num_columns = num_steps + (num_steps - 1)

    fig = plt.figure(figsize=(num_columns * 2.5, 6), dpi=300)

    col_idx = 0
    for _, row in path_df.iterrows():
        if row["step"] == 0:
            # Starting material (reactant of step 0)
            reactant_mol = Chem.MolFromSmiles(row["reactant"])
            reactant_img = Draw.MolToImage(reactant_mol, size=(500, 500))
            ax = fig.add_axes([col_idx / num_columns, 0.3, 1 / num_columns, 0.5])
            ax.imshow(reactant_img)
            ax.axis("off")
            fig.text(
                col_idx / num_columns + 0.5 / num_columns,
                0.35,
                f'QED: {row["qed"]:.3f}',
                ha="center",
                fontsize=12,
            )
            col_idx += 1

        elif row["step"] > 0:
            ax = fig.add_axes(
                [(col_idx - 0.1) / num_columns, 0.3, 1 / num_columns, 0.5]
            )
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
            # Annotate the reaction name close to the arrow
            fig.text(
                (col_idx - 0.1) / num_columns + 0.5 / num_columns,
                0.5,
                row["template"],
                ha="center",
                fontsize=12,
            )
            col_idx += 1

            # Second reactant (if available)
            if pd.notna(row["second_reactant"]):
                second_reactant_mol = Chem.MolFromSmiles(row["second_reactant"])
                second_reactant_img = Draw.MolToImage(
                    second_reactant_mol, size=(500, 500)
                )
                ax = fig.add_axes(
                    [(col_idx - 1) / num_columns, 0.6, 1 / num_columns, 0.4]
                )
                ax.imshow(second_reactant_img)
                ax.axis("off")

                # Plus sign
                fig.text(
                    (col_idx - 1.1) / num_columns + 0.5 / num_columns,
                    0.6,
                    "+",
                    ha="center",
                    fontsize=20,
                )

            # Product
            product_mol = Chem.MolFromSmiles(row["product"])
            product_img = Draw.MolToImage(product_mol, size=(500, 500))
            ax = fig.add_axes([col_idx / num_columns, 0.3, 1 / num_columns, 0.5])
            ax.imshow(product_img)
            ax.axis("off")
            # Annotate the QED value below the product
            fig.text(
                col_idx / num_columns + 0.5 / num_columns,
                0.35,
                f'QED: {row["qed"]:.3f}',
                ha="center",
                fontsize=12,
            )
            col_idx += 1

    # Adjust layout and display the figure
    plt.show()
