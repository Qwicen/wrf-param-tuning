import numpy as np
import pandas as pd
import xarray as xr
import logging
import json
import torch
from sklearn import preprocessing

def get_layers_pd(xr_file, field_name, res):
    data = xr_file[field_name].to_dataframe()
    data = data.reset_index()
    data = data.drop(columns=['Time'])

    def process_frame(fr, save_base_field_name, field_name_from):
        print("process_frame", save_base_field_name, field_name_from)
        nonlocal res

        for ind, row in fr.iterrows():
            west_east = round(row["west_east"])
            south_north = round(row["south_north"])
            save_field_name = f"{save_base_field_name}_{west_east}_{south_north}"
            res[save_field_name] = [row[field_name_from]]

    process_frame(data, field_name, field_name)

def make_raw_feats_from_geodata_pd(xr_file, id):
    res = {}
    res["id"] = id
    def process_layers(field_name):
        nonlocal res
        get_layers_pd(xr_file, field_name, res)

    for i in range(1, 5):
        process_layers(f"OL{i}")
        process_layers(f"OA{i}")

    process_layers("LANDMASK")
    process_layers("HGT_M")
    process_layers("SOILTEMP")
    process_layers("SNOALB")
    process_layers("VAR")
    process_layers("CON")
    process_layers("VAR_SSO")

    res = pd.DataFrame(res, dtype=np.float32)

    return res

def preprocess_feats(frame):
    print("preprocess called")

    def transform_and_ret_scaler(column_name):
        print("transform called")
        column_feats = []

        for i in range(99):
            for j in range(99):
                full_column_name = f"{column_name}_{i}_{j}"
                column_feats = np.append(column_feats, [frame[full_column_name]])

        column_feats = column_feats.reshape((len(column_feats), 1))
        print(f"feats shape {column_feats.shape}")

        scaler = preprocessing.MinMaxScaler().fit(column_feats)
        print("fit returned")

        for i in range(99):
            for j in range(99):
                full_column_name = f"{column_name}_{i}_{j}"
                feats = frame[full_column_name]
                feats = feats.to_numpy().reshape((len(feats), 1))
                frame[full_column_name] = scaler.transform(feats).reshape((len(feats)))

        print("scaler returned")
        return scaler

    return {
        "CON": transform_and_ret_scaler("CON"),
        "HGT_M": transform_and_ret_scaler("HGT_M"),
        "OA1": transform_and_ret_scaler("OA1"),
        "OA2": transform_and_ret_scaler("OA2"),
        "OA3": transform_and_ret_scaler("OA3"),
        "OA4": transform_and_ret_scaler("OA4"),
        "SNOALB": transform_and_ret_scaler("SNOALB"),
        "SOILTEMP": transform_and_ret_scaler("SOILTEMP"),
        "VAR_SSO": transform_and_ret_scaler("VAR_SSO"),
        "VAR": transform_and_ret_scaler("VAR"),
    }

def make_np_layers_from_pd(frame):
    layers = [f"OL{i}" for i in range(1, 5)] + [f"OA{i}" for i in range(1, 5)]
    layers += ["LANDMASK", "HGT_M", "SOILTEMP", "SNOALB", "VAR", "CON", "VAR_SSO"]
    res = np.zeros((len(frame), len(layers), 99, 99))

    current_elem_ind = 0
    for row_ind, row in frame.iterrows():
        print(f"{current_elem_ind} processing")
        for layer_num, layer in enumerate(layers):
            print(f"{layer} processing")
            for west_east in range(99):
                for south_north in range(99):
                    res[current_elem_ind][layer_num][west_east][south_north] = row[f"{layer}_{west_east}_{south_north}"]
        current_elem_ind += 1
    return res

def load_scalers_meta(path='scalers.json'):

    with open(path) as f:
        scalers_meta = json.load(f)

    def load_scaler(scaler_meta):
        scaler = preprocessing.MinMaxScaler()
        scaler.min_ = np.array(scaler_meta["min_"])
        scaler.scale_ = np.array(scaler_meta["scale_"])
        scaler.data_min_ = np.array(scaler_meta["data_min_"])
        scaler.data_max_ = np.array(scaler_meta["data_max_"])
        scaler.data_range_ = np.array(scaler_meta["data_range_"])
        scaler.n_features_in_ = scaler_meta["n_features_in_"]
        scaler.n_samples_seen_ = scaler_meta["n_samples_seen_"]

        return scaler

    res = {scaler_name: load_scaler(scaler_meta) for scaler_name, scaler_meta in scalers_meta.items()}

    return res

def calculate_additional_emb_features(xr_data):
    def get_prepared_data(field_name):
        data = xr_data[field_name].to_dataframe()
        data = data.reset_index()
        data = data.drop(columns=['Time'])
        return data

    def calc_mean_std(np_arr):
        return [np.mean(np_arr), np.std(np_arr)]

    def process_number_month_field(field_name):
        data = get_prepared_data(field_name)

        res = []
        for i in range(12):
            frame = data[data["month"] == i]
            res += calc_mean_std(frame[field_name])
        return res

    def process_landusef():
        data = get_prepared_data("LANDUSEF")

        res = []
        for i in range(21):
             frame = data[data["land_cat"] == i]
             res += [np.mean(frame["LANDUSEF"])]
        return res

    def process_soil_field(field_name):
        data = get_prepared_data(field_name)

        res = []
        for i in range(16):
            frame = data[data["soil_cat"] == i]
            res += [np.mean(frame[field_name])]
        return res

    def process_coord(field_name, max_val):
        data = get_prepared_data(field_name)
        return [(np.mean(data[field_name]) + max_val ) / (2 * max_val)]

    return sum([process_number_month_field("LAI12M"),
                process_number_month_field("ALBEDO12M"),
                process_number_month_field("GREENFRAC"),
                process_landusef(),
                process_soil_field("SOILCBOT"),
                process_soil_field("SOILCTOP"),
                process_coord("XLAT_M", 90),
                process_coord("XLONG_M", 180)],
               [])

def make_geo_emb_from_xr(path, geo_model, scalers, device):
    xr_data = xr.open_dataset(path)
    geo_feats = make_raw_feats_from_geodata_pd(xr_data, 0)

    for scaler_name in scalers:
        for i in range(99):
            for j in range(99):
                full_column_name = f"{scaler_name}_{i}_{j}"
                column_feats = geo_feats[full_column_name]
                column_feats = column_feats.to_numpy().reshape((len(column_feats), 1))
                geo_feats[full_column_name] = scalers[scaler_name].transform(column_feats).reshape((len(column_feats)))

    geo_feats = make_np_layers_from_pd(geo_feats)
    additional_geo_emb = calculate_additional_emb_features(xr_data)

    loader = torch.utils.data.DataLoader(
        geo_feats, batch_size=1, shuffle=False
    )
    assert len(loader) == 1
    for torch_feats in loader:
        geo_emb = geo_model.encoder(torch_feats.float().to(device)).reshape(-1).detach().numpy()

    return np.concatenate((geo_emb, additional_geo_emb))
