# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:56:36 2024

@author: Manoj Kaushik
"""

import pandas as pd
from common_fun import cal_bi_multi_variate_plot_hellinger_distances, normalize_datasets, spectral_angle_mapper_mean

base_path = r"C:\Users\manoj\OneDrive - Indian Institute of Space Science and Technology\5. Project Drone Kolar\Sri_Synthetic\1.Kolar\Data\\"

samples = 1000000 # 10 lakh

# Note: RD => Real Data
RD_healthy_tomato = pd.read_parquet(base_path + r"2.Data_transposed\Parquet\tomato_healthy.parquet")[:samples]
RD_unhealthy_tomato = pd.read_parquet(base_path + r"2.Data_transposed\Parquet\tomato_unhealthy.parquet")[:samples]
RD_Capsicum = pd.read_parquet(base_path + r"2.Data_transposed\Parquet\capsicum.parquet")[:samples]
RD_chilli = pd.read_parquet(base_path + r"2.Data_transposed\Parquet\chilli.parquet")[:samples]
RD_potato = pd.read_parquet(base_path + r"2.Data_transposed\Parquet\potato.parquet")[:samples]
RD_mulberry = pd.read_parquet(base_path + r"2.Data_transposed\Parquet\mulberry.parquet")[:samples]


# Note: SD => Synthetic Data
SD_healthy_tomato = pd.read_parquet(base_path+ r"7.CNN_CVAE\trained on real80 only\tomato_healthy.parquet")[:samples]
SD_unhealthy_tomato = pd.read_parquet(base_path+ r"7.CNN_CVAE\trained on real80 only\tomato_unhealthy.parquet")[:samples]
SD_Capsicum = pd.read_parquet(base_path+ r"7.CNN_CVAE\trained on real80 only\capsicum.parquet")[:samples]
SD_chilli = pd.read_parquet(base_path+ r"7.CNN_CVAE\trained on real80 only\chilli.parquet")[:samples]
SD_potato = pd.read_parquet(base_path+ r"7.CNN_CVAE\trained on real80 only\potato.parquet")[:samples]
SD_mulberry = pd.read_parquet(base_path+ r"7.CNN_CVAE\trained on real80 only\mulberry.parquet")[:samples]


#*********** Preparing the dataset***************
# Healthy_tomato
X1_healthy_tomato = RD_healthy_tomato.drop("label",axis = 1)
Y1_healthy_tomato = RD_healthy_tomato["label"]
X1_healthy_tomato

X2_healthy_tomato = SD_healthy_tomato.drop("label",axis = 1)
Y2_healthy_tomato = SD_healthy_tomato["label"]
X2_healthy_tomato


# Unhealthy_tomato
X1_unhealthy_tomato = RD_unhealthy_tomato.drop("label",axis = 1)
Y1_unhealthy_tomato = RD_unhealthy_tomato["label"]
X1_unhealthy_tomato

X2_unhealthy_tomato = SD_unhealthy_tomato.drop("label",axis = 1)
Y2_unhealthy_tomato = SD_unhealthy_tomato["label"]
X2_unhealthy_tomato

   
# Capsicum
X1_Capsicum = RD_Capsicum.drop("label",axis = 1)
Y1_Capsicum = RD_Capsicum["label"]
X1_Capsicum

X2_Capsicum = SD_Capsicum.drop("label",axis = 1)
Y2_Capsicum = SD_Capsicum["label"]
X2_Capsicum


# Potato
X1_potato = RD_potato.drop("label",axis = 1)
Y1_potato = RD_potato["label"]
X1_potato

X2_potato = SD_potato.drop("label",axis = 1)
Y2_potato = SD_potato["label"]
X2_potato


# Chilli
X1_chilli = RD_chilli.drop("label",axis = 1)
Y1_chilli = RD_chilli["label"]
X1_chilli

X2_chilli = SD_chilli.drop("label",axis = 1)
Y2_chilli = SD_chilli["label"]
X2_chilli
   

# Mulberry
X1_mulberry = RD_mulberry.drop("label",axis = 1)
Y1_mulberry = RD_mulberry["label"]
X1_mulberry

X2_mulberry = SD_mulberry.drop("label",axis = 1)
Y2_mulberry = SD_mulberry["label"]
X2_mulberry


datasets = {
    "Ca": (X1_Capsicum, X2_Capsicum, X1_Capsicum.columns),
    "Ch": (X1_chilli, X2_chilli, X1_chilli.columns),
    "Mu": (X1_mulberry, X2_mulberry, X1_mulberry.columns),
    "Po": (X1_potato, X2_potato, X1_potato.columns),
    "TH": (X1_healthy_tomato, X2_healthy_tomato, X1_healthy_tomato.columns),
    "TU": (X1_unhealthy_tomato, X2_unhealthy_tomato, X1_unhealthy_tomato.columns),
}


# Normalizing the datasets
datasets = normalize_datasets(datasets) 

for dataset_name, (real_data, synthetic_data, columns) in datasets.items():
    print("dataset_name:", dataset_name, "\n", real_data.shape, "\n", synthetic_data.shape, "\n",columns, "\n")


# -----Calculating Uni, multi, Bi variates and plotting-saving the box plot chart------------------------------------------------------
slug="Convolutional CVAE"
cal_bi_multi_variate_plot_hellinger_distances(base_path + r"..\Results\Data_quality\\", datasets, n_bins = 15, slug=slug, save_img=True)


# ---------------------------Apply SAM on mean spectra--------------------------
sam_results = spectral_angle_mapper_mean(datasets)

# Print results
print("\n\nSAM Results (Spectral Angles between Mean Spectra):")
for crop, (X1_mean, X2_mean, columns, angle) in sam_results.items():
    print(f"{crop}: Angle = {angle:.4f} radians")
















