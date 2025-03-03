import os
import sys
import shutil
from tqdm import tqdm


def main():
    src_path, dst_path = sys.argv[1:]
    for cr in tqdm(os.listdir(src_path)):
        for instrument in os.listdir(os.path.join(src_path, cr)):
            for hdf in os.listdir(os.path.join(src_path, cr, instrument)):
                src_hdf = os.path.join(src_path, cr, instrument, hdf)
                dst_hdf = os.path.join(dst_path, cr, instrument, hdf)
                shutil.copy(src_hdf, dst_hdf)


if __name__ == "__main__":
    main()
