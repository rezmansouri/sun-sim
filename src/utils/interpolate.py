import os
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyhdf.SD import SD, SDC


def read_hdf(hdf_path, dataset_names):
    print(f"Reading {hdf_path}...")
    hdf_path = str(hdf_path)
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def save_hdf(hdf_path, data_dict):
    hdf_path = str(hdf_path)
    f = SD(hdf_path, SDC.CREATE | SDC.WRITE)
    for name, array in data_dict.items():
        dset = f.create(name, SDC.FLOAT32, array.shape)
        dset[:] = array
    f.end()


def interpolate_cube(data, x_old, y_old, z_old, x_new, y_new, z_new):
    interp_func = RegularGridInterpolator(
        (x_old, y_old, z_old),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    xg, yg, zg = np.meshgrid(x_new, y_new, z_new, indexing="ij")
    points_new = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=-1)
    data_new_flat = interp_func(points_new)
    data_new = data_new_flat.reshape((len(x_new), len(y_new), len(z_new)))
    return data_new


def process_all(input_root, output_root, grid_hdf_path):
    input_root = Path(input_root)
    output_root = Path(output_root)

    # Read target new grids
    x_new, y_new, z_new = read_hdf(grid_hdf_path, ["fakeDim0", "fakeDim1", "fakeDim2"])

    for vr002_path in input_root.glob("cr*/**/vr002.hdf"):
        print(f"Processing {vr002_path}...")

        # Read original data and grids
        data, x_old, y_old, z_old = read_hdf(
            vr002_path, ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]
        )

        # Interpolate
        data_interp = interpolate_cube(data, x_old, y_old, z_old, x_new, y_new, z_new)

        # Prepare output path
        relative_path = vr002_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save interpolated data
        save_hdf(
            output_path,
            {
                "Data-Set-2": data_interp.astype(np.float32),  # saving as float32
                "fakeDim0": x_new.astype(np.float32),
                "fakeDim1": y_new.astype(np.float32),
                "fakeDim2": z_new.astype(np.float32),
            },
        )

    print("All done!")


process_all(
    input_root="/home/rmansouri1/sun-sim/src/utils/data/hdf",
    output_root="/home/rmansouri1/sun-sim/src/utils/data/shrunk",
    grid_hdf_path="/home/rmansouri1/sun-sim/data/psi_web/6-feb-2025/cr1626/kpo_mas_mas_std_0101/vr002.hdf",
)
