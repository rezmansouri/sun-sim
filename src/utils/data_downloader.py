import os
import pickle5 as pickle
from typing import Dict, List
import urllib3
import requests
import re
from tqdm import tqdm
import pickle
import wget
from datetime import datetime


class Constants:

    FILES_DICT = {
        # "br_r0": "helio/br_r0.hdf",  # ground truth of flux, 1 image
        # "br002": "helio/br002.hdf",  # Simulations of flux, 140 imges
        # "vr_r0": "helio/vr_r0.hdf",  # ground truth of velocity, 1 image
        "vr002": "helio/vr002.hdf",
    }  # Simulations of velocity, 140 images

    FILES_SEQUENCE = [
        #   "br_r0",
        #   "br002",
        #   "vr_r0",
        "vr002"
    ]


class CheckURLs:
    """
    This class checks available simulations and URLs on www.predsci.com website.
    """

    def __init__(
        self,
        start_dir: int,
        end_dir: int,
        start_url: str = "http://www.predsci.com/data/runs/cr",
        end_url: str = "-medium/",
        save_pickle: bool = False,
    ) -> None:
        self.start_dir = start_dir
        self.end_dir = end_dir
        self.start_url = start_url
        self.end_url = end_url
        self.url_dict = None
        self.cr_num_dict = None
        self.save_pickle = save_pickle
        pass

    def collectURLsAll(self) -> Dict:
        usefulurl = {}
        useful_cr_num = {}
        for i in tqdm(range(self.start_dir, self.end_dir)):
            url = self.start_url + str(i) + self.end_url
            # print('---------------------------------------------')
            # print('URL is: ',url)
            deadlink = self.exists(url)
            if not deadlink:
                response = requests.get(url)
                # print(response.text)
                links = re.findall(r'<a[^>]* href="([^"]*)"', response.text)
                run_idx = links.index("/data/runs/")
                sources = links[run_idx + 1 :]
                for src in sources:
                    src = src.strip("/")
                    # print('If url exists: True')
                    # print(url)
                    # usefulurl.append(url)  # print('---------------------------------------------')
                    # print(f'cr{i}')
                    usefulurl[src] = self.dict_add(usefulurl, src, url)
                    useful_cr_num[src] = self.dict_add(useful_cr_num, src, f"cr{i}")
            else:
                # print('If url exists: False')
                dummy = []
        self.url_dict = usefulurl
        self.cr_num_dict = useful_cr_num

        if self.save_pickle:
            save_path = "./data/"
            filename = (
                "url_dict_" + ".pickle"
            )  # datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            filename = save_path + filename
            with open(filename, "wb") as handle:
                pickle.dump(self.url_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.url_dict

    def exists(self, path) -> bool:
        try:
            deadLinkFound = True
            http = urllib3.PoolManager()
            r = http.request("GET", path)
            response = r.status
            if response == 200:
                deadLinkFound = False
                return deadLinkFound
            else:
                return deadLinkFound
        except:
            deadLinkFound = True
            return deadLinkFound

    def dict_add(self, dict_input, key, value) -> List:
        if key not in dict_input:
            return [value]
        else:
            return dict_input[key] + [value]


class DownloadURL:
    """
    This class downloads all URL from given dictionary
    """

    # data/hdf/cr_number/sim_name/files
    def __init__(
        self,
        url: str,
        sim_name: str,
        dir_path: str = "./data",
        dir_name: str = "hdf",
        file_names_dict: Dict = Constants.FILES_DICT,
    ) -> None:
        self.dir_path = dir_path
        self.dir_name = dir_name
        self.url = url.replace("http://", "https://")
        self.sim_name = sim_name
        self.cr_num = self.url.split("/")[-2].split("-")[0]
        self.filenames = file_names_dict

        self.path = (
            self.dir_path
            + "/"
            + self.dir_name
            + "/"
            + self.cr_num
            + "/"
            + self.sim_name
        )

        os.makedirs(self.path, exist_ok=True)

    def get_simulation(self):
        fn_keys = list(self.filenames.keys())
        fn_path_list = {}
        for idx, fn in enumerate(self.filenames):
            final_url = self.url + self.sim_name + "/" + self.filenames[fn]
            path = self.path  # + '/' + fn_keys[idx]
            wget.download(final_url, path)
            file_name = fn + ".hdf"
            file_name = path + "/" + file_name
            os.rename(path + "/" + fn + ".hdf", file_name)
            fn_path_list[fn] = file_name

        return fn_path_list


def main():
    # total available carrington rotations are [1625, 2240]
    download_ = False
    if download_:
        urls = CheckURLs(1625, 2240, save_pickle=True)
        _ = urls.collectURLsAll()

    #### URLs Downloader ###
    url_dict_path = "./data/url_dict_.pickle"
    # loading saved pickle file
    with open(url_dict_path, "rb") as handle:
        urls_dict = pickle.load(handle)

    # list of simulations which will be downloaded
    dwnld_list = [
        "kpo_mas_mas_std_0101",
        # "mdi_mas_mas_std_0101",
        # "hmi_mast_mas_std_0101",
        # "hmi_mast_mas_std_0201",
        # "hmi_masp_mas_std_0201",
        # "mdi_mas_mas_std_0201",
    ]

    # creating a new dict of simulations which needs to be downloaded
    top_n_url_dict = {}
    for sim_name in dwnld_list:
        top_n_url_dict[sim_name] = urls_dict[sim_name]
    for sim_name in top_n_url_dict.keys():
        url_list = top_n_url_dict[sim_name]
        for url_ in url_list:
            dwnld = DownloadURL(url_, sim_name)
            _ = dwnld.get_simulation()
            cr_num = url_.split("/")[-2].split("-")[0]
            print(f"\n {sim_name} - {cr_num} \n")


if __name__ == "__main__":
    main()
