import h5py
import matplotlib.pyplot as plt

file_path = "vr002.h5"

with h5py.File(file_path, 'r') as f:
    def print_structure(name, obj):
        print(name, obj)
    
    print("HDF5 file structure:")
    f.visititems(print_structure)
