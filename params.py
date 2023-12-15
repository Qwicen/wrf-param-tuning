import itertools

def check_restrictions(model_params):
    if model_params["BL_PBL_PHYSICS"] == 5:
        if model_params["SF_SFCLAY_PHYSICS"] not in [1, 5]:
            return False
    elif model_params["BL_PBL_PHYSICS"] == 7:
        if model_params["SF_SFCLAY_PHYSICS"] not in [1, 7]:
            return False
    elif model_params["BL_PBL_PHYSICS"] == 10:
        if model_params["SF_SFCLAY_PHYSICS"] not in [10]:
            return False
    elif model_params["BL_PBL_PHYSICS"] == 11:
        if model_params["SF_SFCLAY_PHYSICS"] not in [1]:
            return False
    return True

def get_all_params():
    available_params = {
        "MP_PHYSICS": [8, 17, 18],
        "RA_LW_PHYSICS": [4],
        "RA_SW_PHYSICS": [4, 5],
        "RADT": [6],
        "SF_SFCLAY_PHYSICS": [1, 5, 7, 10],
        "SF_SURFACE_PHYSICS": [2, 3],
        "SF_URBAN_PHYSICS": [1],
        "BL_PBL_PHYSICS": [5, 7, 10, 11],
        "CU_PHYSICS": [3, 16],
        "DIFF_6TH_OPT": [0, 2],
        "DIFF_OPT": [1, 2],
        "KM_OPT": [4],
    }

    keys = list(available_params)
    all_possible_model_params = [dict(zip(keys, values)) for values in itertools.product(*map(available_params.get, keys))]

    return [model_params for model_params in all_possible_model_params if check_restrictions(model_params)]