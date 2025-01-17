import glob
import logging
import os

# making randomness replicable
import random
import re
import threading
from abc import abstractmethod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.configurable import Configurable
from core.useful_funcs import obs_clean


# region helpers
def convert_key(func):
    def wrapper(self, key, *args, **kwargs):
        if isinstance(key, list):
            fusion_key = key[0]
            for k in key[1:]:
                assert isinstance(k, str)
                fusion_key += k
            key = fusion_key
        return func(self, key, *args, **kwargs)

    return wrapper


class MemoryCache:
    """
    A simple in-memory cache for storing and retrieving data.
    This class provides methods to add, retrieve, and check the existence of data in the cache,
    as well as to clear the cache.

    Attributes:
        cache (dict): A dictionary to store the cached data.
        use_cache (bool): A flag to enable or disable caching.
    """

    def __init__(self, use_cache):
        """
        Initialize the MemoryCache instance.

        Args:
            use_cache (bool): A flag to enable or disable caching.
        """
        self.cache = {}
        self.use_cache = use_cache

    @convert_key
    def add_to_cache(self, key, data):
        """
        Add data to the cache.

        If caching is enabled, the provided data will be stored in the cache using the given key.

        Args:
            key: The key to store the data under. It can be a string or a list of strings
            that will be fused into a single string.
            data: The data to store in the cache.
        """
        if self.use_cache:
            self.cache[key] = data

    @convert_key
    def is_cached(self, key):
        """
        Check if data is present in the cache.

        If caching is enabled, this method will
        return True if the given key is present in the cache, False otherwise.
        If caching is disabled, it will always return False.

        Args:
            key: The key to check in the cache.
            It can be a string or a list of strings that will be fused into a single string.

        Returns:
            bool: True if the key is present in the cache, False otherwise.
        """
        if not self.use_cache:
            return False
        return key in self.cache

    @convert_key
    def get_from_cache(self, key):
        """
        Retrieve data from the cache.

        If caching is enabled, this method will return the data associated
        with the given key from the cache.
        If caching is disabled or the key is not present in the cache, it will return None.

        Args:
            key: The key to retrieve data from the cache.
            It can be a string or a list of strings that will be fused into a single string.

        Returns:
            Any: The cached data associated with the given key,
            or None if the key is not present in the cache or caching is disabled.
        """
        if not self.use_cache:
            return None
        return self.cache[key]

    def clear_cache(self):
        """
        Clear the cache.

        This method will remove all data from the cache.
        """
        self.cache = {}


# endregion

# region Base Dataset


class Dataset(Configurable):
    """
    Base class for datasets.

        This class provides methods to load and cache data from a specified folder.
        Subclasses should define `_get_filename`, `_load_file`, and `__len__` methods.

    To create a custom dataset, follow these steps:

        1. Create a new class that inherits from the `Dataset` class.
        2. Define the `required_keys` class attribute,
        which is a list of required configuration keys for the custom dataset.
        3. Implement the `_get_filename`, `_load_file`, and `__len__` methods in the custom dataset class.

    Example:

    Here's an example of a custom dataset called `CustomDataset`:

    ```python
    class CustomDataset(Dataset):
        required_keys = ['custom_key']

        def __init__(self, config_data, use_cache=True, **kwargs):
            super().__init__(config_data, use_cache)
        # self.my_custom_key = config_data['my_custom_key']
        # my_custom_key is automatically set as an attribute on the instance by the Configurable base class

        def _get_filename(self, index):
            return os.path.join(self.data_folder, f"custom_file_{index}.npy")

        def _load_file(self, file_path):
            return np.load(file_path)

        def __len__(self):
            return 1000
    ```

    In this example, `CustomDataset` has a custom attribute `custom_attribute`
    and requires a configuration key called `custom_key`.
    The `_get_filename`, `_load_file`, and `__len__` methods are implemented
    to define the behavior for loading and accessing the data.
    """

    required_keys = [
        "data_folder",
    ]

    def __init__(self, config_data, use_cache=True, **kwargs):
        """
        Initialize the Dataset instance.

        Args:
            config_data (dict): The configuration data for the dataset.
            use_cache (bool, optional): A flag to enable or disable caching. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.cache = MemoryCache(use_cache)
        self.load_data_semaphore = threading.Semaphore()

    @abstractmethod
    def _get_filename(self, index):
        """
        Get the filename for the specified index.

        This method should be implemented by subclasses to provide the logic
        for obtaining the filename based on the index.

        Args:
            index (int): The index of the file.

        Returns:
            str: The filename.
        """
        pass

    @abstractmethod
    def _load_file(self, file_path):
        """
        Load the data from the specified file path.

        This method should be implemented by subclasses to provide the logic for loading the data from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            Any: The loaded data.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
                Get the length of the dataset.
        self.batch_size

                Returns:
                    int: The length of the dataset.
        """
        pass

    def _load(self, file_path):
        """
        Load the data from the specified file path.

        If the data is not cached, it will be loaded and stored in the cache.
        If the data is cached, it will be retrieved from the cache.

        Args:
            file_path (str): The path to the file.

        Returns:
            Any: The data.
        """
        if not self.cache.is_cached(file_path):
            data = self._load_file(file_path)
            self.cache.add_to_cache(file_path, data)
        else:
            data = self.cache.get_from_cache(file_path)
        return data

    def is_dataset_cached(self):
        """
        Check if the entire dataset is cached.

        Returns:
            bool: True if the entire dataset is cached, False otherwise.
        """
        for idx in range(len(self)):
            file_path = self._get_filename(idx)
            if not self.cache.is_cached(file_path):
                return False
        return True

    def get_all_data(self):
        """
        Get all data from the dataset.

        If the data is not cached, it will be loaded and stored in the cache.
        If the data is cached, it will be retrieved from the cache.

        Returns:
            np.ndarray: The concatenated  data from the entire dataset.
        """
        all_data = []
        if not self.is_dataset_cached():
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Collecting uncached data"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = self._load(file_path)
                    all_data.append(data)
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        else:
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Getting data from cache"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = self.cache.get_from_cache(file_path)
                    all_data.append(data)
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        return np.concatenate(all_data, axis=0)

    def __getitem__(self, items):
        """
        Get the data for the specified index or indices.

        Args:
            items: The index or indices of the data to retrieve.

        Returns:
            Any: The data.
        """
        file_path = self._get_filename(items)
        data = self._load(file_path)
        return data

    def _get_full_path(self, filename, extension=".npy"):
        """
        Get the full path of a file given its filename and extension.

        Args:
            filename (str): The filename.
            extension (str, optional): The file extension. Defaults to ".npy".

        Returns:
            str: The full path of the file.
        """
        return os.path.join(self.data_folder, f"{filename}{extension}")


# endregion

# region custom datasets


class DateDataset(Dataset):
    required_keys = ["data_folder", "crop_indices"]

    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)
        self.df0 = pd.read_csv(
            os.path.join(config_data["path_to_csv"], config_data["csv_file"])
        )
        df_extract = self.df0[
            (self.df0["Date"] >= config_data["date_start"])
            & (self.df0["Date"] < config_data["date_end"])
        ]
        self.df0 = self.df0
        self.liste_dates = df_extract["Date"].unique().tolist()
        self.liste_dates = self.liste_dates[0 : config_data["number_of_dates"]]
        self.liste_dates_repl = [
            date_string.replace("T21:00:00Z", "") for date_string in self.liste_dates
        ]
        self.liste_dates_rep = [
            item
            for item in self.liste_dates_repl
            for _ in range(config_data["Lead_Times"])
        ]

    def _get_filename(self, index):
        raise NotImplementedError("Subclasses must implement this method.")

    def _load_file(self, file_path):
        raise NotImplementedError("Subclasses must implement this method.")

    def __len__(self):
        return len(self.liste_dates_rep)


class ObsDataset(DateDataset):
    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)
        self.filename_format = config_data.get(
            "filename_format", "obs{date}_{formatted_index}"
        )

    def _get_filename(self, index):
        format_variables = [
            var.strip("}{") for var in re.findall(r"{(.*?)}", self.filename_format)
        ]
        kwargs = {}

        real_hour = self.start_time + (index % self.Lead_Times + 1) * self.dh

        if "formatted_index" in format_variables:
            format_variables.remove("formatted_index")
            formatted_index = real_hour % 24
            kwargs = {"formatted_index": formatted_index}

        if "date" in format_variables:
            format_variables.remove("date")
            date = self.liste_dates_rep[index]
            date_index = int(np.floor(real_hour / 24.0))
            date_0 = datetime.strptime(date, "%Y-%m-%d")
            next_date_1 = date_0 + timedelta(days=1)
            next_date_2 = date_0 + timedelta(days=2)
            date_1 = next_date_1.strftime("%Y-%m-%d")
            date_2 = next_date_2.strftime("%Y-%m-%d")
            dates = [date, date_1, date_2]
            kwargs = kwargs | {"date": dates[date_index].replace("-", "")}

        kwargs = kwargs | {var: getattr(self, var, "") for var in format_variables}

        return self._get_full_path(self.filename_format.format(**kwargs))

    def _load_file(self, file_path):
        return obs_clean(np.load(file_path).astype(np.float32), self.crop_indices)

    def get_all_data(self):
        all_data = []
        if not self.is_dataset_cached():
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Collecting uncached data"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = self._load(file_path)
                    all_data.append(data[np.newaxis, :, :, :])
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        else:
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Getting data from cache"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = self.cache.get_from_cache(file_path)
                    all_data.append(data[np.newaxis, :, :, :])
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        return np.concatenate(all_data, axis=0)


class FakeDataset(DateDataset):
    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)

        self.filename_format = config_data.get(
            "filename_format",
            "genFsemble_{date}_{formatted_index}_{inv_step}_{cond_members}_{N_ens}",
        )

    def _get_filename(self, index):
        format_variables = [
            var.strip("}{") for var in re.findall(r"{(.*?)}", self.filename_format)
        ]
        kwargs = {}

        if "formatted_index" in format_variables:
            format_variables.remove("formatted_index")
            formatted_index = (index % self.Lead_Times + 1) * self.dh
            kwargs = {"formatted_index": formatted_index}

        if "date" in format_variables:
            format_variables.remove("date")
            date = self.liste_dates_rep[index]
            kwargs = kwargs | {"date": date}

        kwargs = kwargs | {var: getattr(self, var, "") for var in format_variables}

        return self._get_full_path(self.filename_format.format(**kwargs))

    def _load_file(self, file_path):
        return np.load(file_path).astype(np.float32)


class RealDataset(DateDataset):
    def _get_filename(self, index):
        date = self.liste_dates_rep[index]
        names = self.df0[
            (self.df0["Date"] == f"{date}T21:00:00Z")
            & (self.df0["LeadTime"] == (index % self.Lead_Times + 1) * self.dh - 1)
        ]["Name"].to_list()
        file_names = [self._get_full_path(name) for name in names]
        return file_names

    def _load_file(self, file_path):
        arrays = [
            np.expand_dims(np.load(file_name).astype(np.float32), axis=0)
            for file_name in file_path
        ]
        return np.concatenate(arrays, axis=0)


class RandomDataset(Dataset):
    required_keys = [
        "data_folder",
        "config",
        "crop_indices",
        "filename_format",
        "maxNsamples",
        "file_size",
    ]

    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)
        self.filename_format = config_data.get(
            "filename_format", "_Fsemble_{step}_{index}"
        )
        self.data_folder = config_data["data_folder"]
        format_variables = [
            var.strip("}{") for var in re.findall(r"{(.*?)}", self.filename_format)
        ]
        kwargs = {}
        kwargs = kwargs | {
            var: getattr(self, var, "") for var in format_variables if var != "index"
        }
        kwargs["index"] = "*"
        self.filelist = glob.glob(
            os.path.join(self.data_folder, self.filename_format.format(**kwargs))
        )
        random.shuffle(self.filelist)
        self.filelist = self.filelist[
            : int(config_data["maxNsamples"]) // config_data["file_size"]
        ]

    def _get_full_path(self, filename, extension=".npy"):
        return os.path.join(self.data_folder, f"{filename}{extension}")

    def _get_filename(self, index):
        return self.filelist[index]

    def _load_file(self, file_path):
        return np.load(file_path).astype(np.float32)

    def __len__(self):
        return len(self.filelist)

    def get_all_data(self):
        all_data = []
        if not self.is_dataset_cached():
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Collecting uncached data"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = (
                        self._load(file_path)[np.newaxis, :, :, :]
                        if self.file_size == 1
                        else self._load(file_path)
                    )
                    all_data.append(data)
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        else:
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Getting data from cache"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = (
                        self.cache.get_from_cache(file_path)[np.newaxis, :, :, :]
                        if self.file_size == 1
                        else self.cache.get_from_cache(file_path)
                    )
                    all_data.append(data)
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        return np.concatenate(all_data, axis=0)


class MixDataset(DateDataset):
    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)

        self.filename_format = config_data.get(
            "filename_format",
            "genFsemble_{date}_{formatted_index}_{inv_step}_{cond_members}_{N_ens}",
        )
        self.N_real_mb = int(
            config_data.get("real_proportion", 0.0) * config_data["N_ens"]
        )
        if (
            self.N_real_mb > 16
        ):  # hard constraint here since AROME data only have 16 members at most
            raise Warning(
                f"input proportion of real mbs : {config_data['real_proportion']} total {config_data['N_ens']} mbs,\
            but AROME ensemble only have 16 members. Capping real members number to 16."
            )
            self.N_real_mb = 16
        self.N_fake_mb = config_data["N_ens"] - self.N_real_mb
        self.real_data_folder = config_data["real_dataset_config"]["data_folder"]
        self.real_var_indices = config_data["real_dataset_config"]["real_var_indices"]

    def _get_real_full_path(self, filename, extension=".npy"):
        return os.path.join(self.real_data_folder, f"{filename}{extension}")

    def _get_fake_full_path(self, filename, extension=".npy"):
        return os.path.join(self.data_folder, f"{filename}{extension}")

    def _get_real_filename(self, index):
        date = self.liste_dates_rep[index]
        names = self.df0[
            (self.df0["Date"] == f"{date}T21:00:00Z")
            & (self.df0["LeadTime"] == (index % self.Lead_Times + 1) * self.dh - 1)
        ]["Name"].to_list()
        file_names = [self._get_real_full_path(name) for name in names]
        return file_names

    def _get_fake_filename(self, index):
        format_variables = [
            var.strip("}{") for var in re.findall(r"{(.*?)}", self.filename_format)
        ]
        kwargs = {}

        if "formatted_index" in format_variables:
            format_variables.remove("formatted_index")
            formatted_index = (index % self.Lead_Times + 1) * self.dh
            kwargs = {"formatted_index": formatted_index}

        if "date" in format_variables:
            format_variables.remove("date")
            date = self.liste_dates_rep[index]
            kwargs = kwargs | {"date": date}

        kwargs = kwargs | {var: getattr(self, var, "") for var in format_variables}

        return self._get_fake_full_path(self.filename_format.format(**kwargs))

    def _get_filename(self, index):
        return {
            "real": self._get_real_filename(index),
            "fake": self._get_fake_filename(index),
        }

    def _load_real_file(self, file_path):
        arrays = [
            np.expand_dims(np.load(file_name).astype(np.float32), axis=0)
            for file_name in file_path
        ]
        return np.concatenate(arrays, axis=0)

    def _load_fake_file(self, file_path):
        return np.load(file_path).astype(np.float32)

    def _load_file(self, file_path):
        real_file = self._load_real_file(file_path["real"])
        fake_file = self._load_fake_file(file_path["fake"])
        sample = np.concatenate((real_file, fake_file), axis=0)
        return sample

    def _load(self, file_path):
        if not self.cache.is_cached(file_path["real"]):
            data = self._load_file(file_path)
            self.cache.add_to_cache(file_path["real"], data)
        else:
            data = self.cache.get_from_cache(file_path["real"])
        return data


class ModDataset(DateDataset):
    """
    dataset where fake data are modified by another source of fake data in a preselected way
    Allows for debiasing in particular
    """

    required_keys = ["data_folder", "mod_data_folder", "filename_mod_format"]

    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache, **kwargs)

        self.filename_format = config_data.get(
            "filename_format",
            "genFsemble_{date}_{formatted_index}_{inv_step}_{cond_members}_{N_ens}",
        )
        self.filename_mod_format = config_data.get(
            "filename_mod_format",
            "invertFsemble_{date}_{formatted_index}_{inv_step}_{cond_members}_{N_ens}",
        )

    def _get_fake_filename(self, index):
        format_variables = [
            var.strip("}{") for var in re.findall(r"{(.*?)}", self.filename_format)
        ]
        kwargs = {}

        if "formatted_index" in format_variables:
            format_variables.remove("formatted_index")
            formatted_index = (index % self.Lead_Times + 1) * self.dh
            kwargs = {"formatted_index": formatted_index}

        if "date" in format_variables:
            format_variables.remove("date")
            date = self.liste_dates_rep[index]
            kwargs = kwargs | {"date": date}

        kwargs = kwargs | {var: getattr(self, var, "") for var in format_variables}

        return self._get_full_path(self.filename_format.format(**kwargs))

    def _get_full_path(self, filename, extension=".npy", mod=False):
        if mod:
            return os.path.join(self.mod_data_folder, f"{filename}{extension}")
        return os.path.join(self.data_folder, f"{filename}{extension}")

    def _get_mod_filename(self, index):
        format_variables = [
            var.strip("}{") for var in re.findall(r"{(.*?)}", self.filename_mod_format)
        ]
        kwargs = {}

        if "formatted_index" in format_variables:
            format_variables.remove("formatted_index")
            formatted_index = (index % self.Lead_Times + 1) * self.dh
            kwargs = {"formatted_index": formatted_index}

        if "date" in format_variables:
            format_variables.remove("date")
            date = self.liste_dates_rep[index]
            kwargs = kwargs | {"date": date}

        kwargs = kwargs | {var: getattr(self, var, "") for var in format_variables}

        return self._get_full_path(self.filename_mod_format.format(**kwargs), mod=True)

    def _get_filename(self, index):
        fake_filename = self._get_fake_filename(index)
        mod_filename = self._get_mod_filename(index)
        return {"fake_path": fake_filename, "mod_path": mod_filename}

    def _load(self, file_path):
        if not self.cache.is_cached(file_path["fake_path"]):
            data = self._load_file(file_path)
            self.cache.add_to_cache(file_path["fake_path"], data)
        else:
            data = self.cache.get_from_cache(file_path["fake_path"])
        return data

    def _load_file(self, file_path):
        return {
            "fake": np.load(file_path["fake_path"]).astype(np.float32),
            "mod": np.load(file_path["mod_path"]).astype(np.float32),
        }

    def get_all_data(self):
        all_data_fake = []
        all_data_mod = []
        if not self.is_dataset_cached():
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Collecting uncached data"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = self._load(file_path["fake_path"])
                    all_data_fake.append(data["fake"])
                    all_data_mod.append(data["mod"])
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        else:
            for idx in tqdm(
                range(len(self)), desc=f"{self.name} : Getting data from cache"
            ):
                try:
                    file_path = self._get_filename(idx)
                    data = self.cache.get_from_cache(file_path["fake_path"])
                    all_data_fake.append(data["fake"])
                    all_data_mod.append(data["mod"])
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        return np.concatenate(all_data_fake,axis=0), np.concatenate(all_data_mod,axis=0)

class DiffDataset(Dataset):
    r"""
    Dataset where data are temporal difference (absolute or not) of meteorological fields.
    Delta_X(t,dh) = X(t + dh) - X(t) ; Where X(t) is a meteorological field at time step t.
    """
    required_keys = ['data_folder', 'preprocessor_config', 'crop_indices', 'temporal_difference_type']

    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)
        self.df0 = pd.read_csv(os.path.join(config_data['path_to_csv'], config_data['csv_file']))
        df_extract = self.df0[
            (self.df0['Date'] >= config_data['date_start']) & (self.df0['Date'] < config_data['date_end'])]
        self.df0 = self.df0
        self.liste_dates = df_extract['Date'].unique().tolist()
        self.liste_dates = self.liste_dates[0:config_data['number_of_dates']]
        self.liste_dates_repl = [date_string.replace('T21:00:00Z', '') for date_string in self.liste_dates]
        self.liste_dates_rep = [item for item in self.liste_dates_repl for _ in range(config_data['Lead_Times'])]

    def is_dataset_cached(self):
        """
        Check if the entire dataset is cached.

        Returns:
            bool: True if the entire dataset is cached, False otherwise.
        """
        for idx in range(len(self)):
            file_path_t, file_path_t_next = self._get_filename(idx)
            if not self.cache.is_cached(file_path_t):
                return False
            if not self.cache.is_cached(file_path_t_next):
                return False
        return True
    
    def __getitem__(self, items):
        """
        Get the preprocessed data for the specified index or indices.

        Args:
            items: The index or indices of the data to retrieve.

        Returns:
            Any: The preprocessed data.
        """
        file_path_t, file_path_t_next = self._get_filename(items)
        data_t = self._load(file_path_t)
        data_t_next = self._load(file_path_t_next)
        if self.temporal_difference_type == 'absolute':
            return np.abs(np.array(data_t_next-data_t, dtype=np.float32))/self.dh
        elif self.temporal_difference_type == 'simple':
            return np.array(data_t_next-data_t, dtype=np.float32)/self.dh
        else :
            raise NotImplementedError
        
    def __len__(self):
        return len(self.liste_dates_rep)
    
    def _get_filename(self, index):
        date = self.liste_dates_rep[index]
        # Selecting Leadtime ids : 

        # index % self.Lead_Times + 1 : allow to iterate over Leadtimes no matter what index is 
        # ex : if index is [0, 1, 2, 3, 4, 5, 6] and Lead_Times is 3 then the list obtain will be [1, 2, 3, 1, 2, 3, 1]
        # But we also want to jump over leadtime with a certain step
        # (index % self.Lead_Times + 1) * self.dh - 1
        # ex : like before if we have dh=3 then we will obtain the list [2, 5, 8, 2, 5, 8, 2]

        names_t = self.df0[
            (self.df0['Date'] == f"{date}T21:00:00Z") & (
                    self.df0['LeadTime'] == (index % self.Lead_Times + 1) * self.dh - 1 )][
            'Name'].to_list()

        # We select the next time step by adding 1 step to leadtime id
        names_t_next = self.df0[
            (self.df0['Date'] == f"{date}T21:00:00Z") & (
                    self.df0['LeadTime'] == (index % self.Lead_Times + 1) * self.dh)][
            'Name'].to_list()

        # print(f'Comparing : t={(index % self.Lead_Times + 1) * self.dh - 1} with t={(index % self.Lead_Times + 1) * self.dh}')
        file_names_t = [self._get_full_path(name) for name in names_t]
        file_names_t_next = [self._get_full_path(name) for name in names_t_next]
        return file_names_t, file_names_t_next

    def _load_file(self, file_path):
        arrays = [np.expand_dims(np.load(file_name), axis=0) for file_name in file_path]
        return np.concatenate(arrays, axis=0)
    
    def get_all_data(self):
        """
        Get all data from the dataset.

        If the data is not cached, it will be loaded, preprocessed, and stored in the cache.
        If the data is cached, it will be retrieved from the cache.

        Returns:
            np.ndarray: The concatenated preprocessed data from the entire dataset.
        """
        all_data = []
        if not self.is_dataset_cached():
            for idx in tqdm(range(len(self)), desc=f"{self.name} : Collecting uncached data"):
                try:
                    file_path_t, file_path_t_next = self._get_filename(idx)
                    data_t = self._load(file_path_t)
                    data_t_next = self._load(file_path_t_next)
                    if self.temporal_difference_type == 'absolute':
                        all_data.append(np.abs(np.array(data_t_next-data_t, dtype=np.float32))/self.dh)
                    elif self.temporal_difference_type == 'simple':
                        all_data.append(np.array(data_t_next-data_t, dtype=np.float32)/self.dh)
                    else :
                        raise NotImplementedError
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        else:
            for idx in tqdm(range(len(self)), desc=f"{self.name} : Getting data from cache"):
                try:
                    file_path_t, file_path_t_next = self._get_filename(idx)
                    data_t = self.cache.get_from_cache(file_path_t)
                    data_t_next = self.cache.get_from_cache(file_path_t_next)
                    if self.temporal_difference_type == 'absolute':
                        all_data.append(np.abs(np.array(data_t_next-data_t, dtype=np.float32))/self.dh)
                    elif self.temporal_difference_type == 'simple':
                        all_data.append(np.array(data_t_next-data_t, dtype=np.float32)/self.dh)
                    else :
                        raise NotImplementedError
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        return np.concatenate(all_data, axis=0)


class DiffDateDataset(DateDataset):
    def __init__(self, config_data, use_cache=True, **kwargs):
        super().__init__(config_data, use_cache)

        self.filename_format = config_data.get('filename_format',
                                               "genFsemble_{date}_{formatted_index}_{inv_step}_{cond_members}_{N_ens}")

    def _get_filename(self, index):
        format_variables = [var.strip('}{') for var in re.findall(r'{(.*?)}', self.filename_format)]
        kwargs_t = {}
        kwargs_t_next = {}

        if 'formatted_index' in format_variables:
            format_variables.remove('formatted_index')
            formatted_index = (index % self.Lead_Times + 1) * self.dh
            kwargs_t = {'formatted_index': formatted_index}
            kwargs_t_next = {'formatted_index': formatted_index + self.dh}
            # print(f'Comparing time steps {formatted_index} and {formatted_index+self.dh}')

        if 'date' in format_variables:
            format_variables.remove('date')
            date = self.liste_dates_rep[index]
            kwargs_t = kwargs_t | {'date': date}
            kwargs_t_next = kwargs_t_next | {'date': date}

        kwargs_t = kwargs_t | {var: getattr(self, var, '') for var in format_variables}
        kwargs_t_next = kwargs_t_next | {var: getattr(self, var, '') for var in format_variables}

        return self._get_full_path(
            self.filename_format.format(**kwargs_t)
        ), self._get_full_path(
            self.filename_format.format(**kwargs_t_next)
        )

    def is_dataset_cached(self):
        """
        Check if the entire dataset is cached.

        Returns:
            bool: True if the entire dataset is cached, False otherwise.
        """
        for idx in range(len(self)):
            file_path_t, file_path_t_next = self._get_filename(idx)
            if not self.cache.is_cached(file_path_t):
                return False
            if not self.cache.is_cached(file_path_t_next):
                return False
        return True
    
    def __getitem__(self, items):
        """
        Get the preprocessed data for the specified index or indices.

        Args:
            items: The index or indices of the data to retrieve.

        Returns:
            Any: The preprocessed data.
        """
        file_path_t, file_path_t_next = self._get_filename(items)
        data_t = self._load(file_path_t)
        data_t_next = self._load(file_path_t_next)
        if self.temporal_difference_type == 'absolute':
            return np.abs(np.array(data_t_next-data_t, dtype=np.float32))/self.dh
        elif self.temporal_difference_type == 'simple':
            return np.array(data_t_next-data_t, dtype=np.float32)/self.dh
        else :
            raise NotImplementedError
        
    def __len__(self):
        return len(self.liste_dates_rep)
    
    def get_all_data(self):
        """
        Get all data from the dataset.

        If the data is not cached, it will be loaded, preprocessed, and stored in the cache.
        If the data is cached, it will be retrieved from the cache.

        Returns:
            np.ndarray: The concatenated preprocessed data from the entire dataset.
        """
        all_data = []
        if not self.is_dataset_cached():
            for idx in tqdm(range(len(self)), desc=f"{self.name} : Collecting uncached data"):
                try:
                    file_path_t, file_path_t_next = self._get_filename(idx)
                    data_t = self._load(file_path_t)
                    data_t_next = self._load(file_path_t_next)
                    if self.temporal_difference_type == 'absolute':
                        all_data.append(np.abs(np.array(data_t_next-data_t, dtype=np.float32))/self.dh)
                    elif self.temporal_difference_type == 'simple':
                        all_data.append(np.array(data_t_next-data_t, dtype=np.float32)/self.dh)
                    else :
                        raise NotImplementedError
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        else:
            for idx in tqdm(range(len(self)), desc=f"{self.name} : Getting data from cache"):
                try:
                    file_path_t, file_path_t_next = self._get_filename(idx)
                    data_t = self.cache.get_from_cache(file_path_t)
                    data_t_next = self.cache.get_from_cache(file_path_t_next)
                    if self.temporal_difference_type == 'absolute':
                        all_data.append(np.abs(np.array(data_t_next-data_t, dtype=np.float32))/self.dh)
                    elif self.temporal_difference_type == 'simple':
                        all_data.append(np.array(data_t_next-data_t, dtype=np.float32)/self.dh)
                    else :
                        raise NotImplementedError
                except FileNotFoundError as e:
                    logging.warning(f"FileNotFound {e}, continuing")
        return np.concatenate(all_data, axis=0)
    
    def _load_file(self, file_path):
        return np.load(file_path)
