import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

class CensysDataset(Dataset):
    def __init__(self, filepath):

        # Set up lists to work with data
        self.country_code_list = []
        self.latitude_list = []
        self.longitude_list = []
        self.asn_list = []
        # self.bgp_prefix_list = []                 # These two have been removed since including IPs in a machine learning dataset isn't very helpful
        # self.bgp_prefix_mask_list = []
        self.country_code_asn_list = []
        self.service_names_list_list = []
        self.vendor_list = []
        self.product_list = []
        self.version_list = []

        self.country_code_set = set()
        self.country_code_asn_set = set()
        self.service_set = set()
        self.vendor_set = set()
        self.product_set = set()

        # Parse data; converts latitude, longitude, asn, and version into numerical fields (more details in the parse_data method)
        self.__parsedata(filepath)

        # Convert numerical fields (latitude, longitude, asn) to pytorch, then normalize them by dividing by the largest value
        self.latitude_list = torch.FloatTensor(self.latitude_list)
        self.longitude_list = torch.FloatTensor(self.longitude_list)
        self.asn_list = torch.FloatTensor(self.asn_list)
        self.version_list = torch.FloatTensor(self.version_list)

        self.latitude_list = self.latitude_list / torch.max(self.latitude_list)
        self.longitude_list = self.longitude_list / torch.max(self.longitude_list)
        self.asn_list = self.asn_list / torch.max(self.asn_list)
        self.version_list = self.version_list / torch.max(self.version_list)


        # Set up label encoder for each of the string fields (country_code, country_code_asn. service_names_list, vendor, and product)
        self.cc_le = LabelEncoder()
        self.cc_asn_le = LabelEncoder()
        self.service_le = LabelEncoder()
        self.vendor_le = LabelEncoder()
        self.product_le = LabelEncoder()

        self.cc_le.fit(list(self.country_code_set))
        self.num_cc = len(self.cc_le.classes_) - 1

        self.cc_asn_le.fit(list(self.country_code_asn_set))
        self.num_cc_asn = len(self.cc_asn_le.classes_) - 1

        self.service_le.fit(list(self.service_set))
        self.num_classes = len(self.service_le.classes_)

        self.vendor_le.fit(list(self.vendor_set))
        self.num_vend = len(self.vendor_le.classes_) - 1

        self.product_le.fit(list(self.product_set))
        self.num_prod = len(self.product_le.classes_) - 1
            
            

    def __len__(self):
        return len(self.service_names_list_list)
    
    def __getitem__(self, idx):
        # Retrieve country code, convert it to a float, and normalize it by dividing by the total number of country codes
        country = self.country_code_list[idx]
        country = float(self.cc_le.transform([country])[0]) / self.num_cc

        # Retrieve latitude, longitude, asn, and version, already normalized
        latitude = self.latitude_list[idx]
        longitude = self.longitude_list[idx]
        asn = self.asn_list[idx]
        version = self.version_list[idx]

        # Retrieve asn country code, convert it to a float, and normalize it by dividing by the total number of asn country codes
        country_asn = self.country_code_asn_list[idx]
        country_asn = float(self.cc_asn_le.transform([country_asn])[0]) / self.num_cc_asn

        # Do the same for vendor, and product
        vendor = self.vendor_list[idx]
        vendor = float(self.vendor_le.transform([vendor])[0]) / self.num_vend

        product = self.product_list[idx]
        product = float(self.product_le.transform([product])[0]) / self.num_prod

        # Create feature vector from the above
        features = torch.FloatTensor([country, latitude, longitude, asn, country_asn, vendor, product, version])

        # Retrieve service list and transform it to categorical labels
        services = self.service_names_list_list[idx]
        services = self.service_le.transform(services)
        multihot = torch.zeros(self.num_classes)
        multihot[services] = 1

        return features, multihot
    
    def __parsedata(self, filepath):
        df = pd.read_json(filepath, lines=True)
        for cc, lat, long, asn, bgp_prefix, cc_asn, snl, vend, prod, ver in zip(df["country_code"],
                                                                                df["latitude"],
                                                                                df["longitude"],
                                                                                df["asn"],
                                                                                df["bgp_prefix"],
                                                                                df["country_code_1"],
                                                                                df["service_names_list"],
                                                                                df["vendor"],
                                                                                df["product"],
                                                                                df["version"]):
            # Read country code
            if pd.isna(cc):
                cc = "None"
            self.country_code_list.append(cc)
            self.country_code_set.add(cc)

            # Read latitude
            if pd.isna(lat):
                lat = float(-1)
            self.latitude_list.append(lat)
            
            # Read longitude
            if pd.isna(long):
                long = float(-1)
            self.longitude_list.append(long)

            # Read asn, convert to integer value
            if pd.isna(asn):
                asn = -1
            else:
                asn = int(asn)
            self.asn_list.append(asn)

            # Read asn country code
            if pd.isna(cc_asn):
                cc_asn = "None"
            self.country_code_asn_list.append(cc_asn)
            self.country_code_asn_set.add(cc_asn)

            # You get it...
            if pd.isna(snl)[0]:
                snl = []
            self.service_names_list_list.append(snl)
            for s in snl:
                self.service_set.add(s)

            if pd.isna(vend):
                vend = "None"
            self.vendor_list.append(vend)
            self.vendor_set.add(vend)

            if pd.isna(prod):
                prod = "None"
            self.product_list.append(prod)
            self.product_set.add(prod)

            # Remove non-decimal characters from version name, remove all decimals after the first
            if pd.isna(ver) or not bool(re.search(r'\d', ver)):
                ver = float(-1)
            else:
                non_decimal = re.compile(r'[^\d.]+')
                version = non_decimal.sub('', ver)
                version_processing = version.split(".")
                version_1 = ".".join(version_processing[:2])
                version_processing = version_processing[2:]
                version_processing.insert(0, version_1)
                final_version = "".join(version_processing)
                ver = float(final_version)
            self.version_list.append(ver)

    def get_num_classes(self):
        return self.num_classes