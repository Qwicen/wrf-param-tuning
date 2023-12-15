import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import features
import argparse
from autoencoder import Autoencoder
from optimizer import WrfParamsOptimizer
from templates.render_templates import render_wps_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--geo-path', type=str, default='/home/wrfuser/WPS/geo_em.d01.nc')
    parser.add_argument('--ae-path', type=str, default='/home/wrfuser/pipeline/models/emb_middle_model_params.pt')
    parser.add_argument("--cb-path", type=str, default='/home/wrfuser/pipeline/models/catboost_model')
    parser.add_argument("--scaler-path", type=str, default='/home/wrfuser/pipeline/models/scalers.json')
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(args.ae_path, map_location=device))
    scalers = features.load_scalers_meta(args.scaler_path)
    geo_feats = features.make_geo_emb_from_xr(args.geo_path, model, scalers, device=device)
    optimizer = WrfParamsOptimizer(geo_feats)
    optimizer.load_model(args.cb_path)
    all_params, all_predicts, all_uncertanties = optimizer.get_all_predicts()
    optimized_params = all_params[np.argmin(all_predicts)]
    print(f"optimized_params = {optimized_params}")

