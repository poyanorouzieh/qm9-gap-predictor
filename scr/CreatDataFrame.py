import pathlib

import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------

file_path = pathlib.Path(__file__).resolve()
data_dir = file_path.parent.parent / "dataset" / "133660_curatedQM9_outof_133885"

file = list(data_dir.glob("dsgdb9nsd_*.xyz"))

data_list = []

# ----------------------------------------------------------------------------------------------


for path in tqdm(file):
    try:
        with open(path, "r") as iteration:
            lines = iteration.readlines()
            props = lines[1].split()

            if props[0] != "gdb":
                raise ValueError("format mismatch")

            row = {
                "source_file": path.name,
                "na": int(lines[0].strip()),
                "index": int(props[1]),
                "A": float(props[2]),
                "B": float(props[3]),
                "C": float(props[4]),
                "mu": float(props[5]),
                "alpha": float(props[6]),
                "homo": float(props[7]),
                "lumo": float(props[8]),
                "gap": float(props[9]),
                "r2": float(props[10]),
                "zpve": float(props[11]),
                "U0": float(props[12]),
                "U": float(props[13]),
                "H": float(props[14]),
                "G": float(props[15]),
                "Cv": float(props[16]),
            }

        data_list.append(row)


    except Exception as e:  # noqa: BLE001
        print("skip:", path.name, e)

# ----------------------------------------------------------------------------------------------

df = pd.DataFrame(data_list)
output_path = file_path.parent.parent / "qm9_dataset_full_data_v1.csv"
df.to_csv(output_path, index=False)

# ----------------------------------------------------------------------------------------------

print("finished :", df.shape)
