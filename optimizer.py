import copy
import numpy as np
import pandas as pd
import catboost
import matplotlib.pyplot as plt
from params import get_all_params
import features

class WrfParamsOptimizer():
    def __init__(self, geo_emb_cache, use_uncertainty=False):
        self.use_uncertainty = use_uncertainty
        self.geo_scaler = None
        self.regressor = self._create_regressor()
        self.filtered_columns = ["MP_PHYSICS",  "RA_LW_PHYSICS", "RA_SW_PHYSICS", "RADT", "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS", "SF_URBAN_PHYSICS", "BL_PBL_PHYSICS", "CU_PHYSICS", "DIFF_6TH_OPT", "DIFF_OPT", "KM_OPT"]
        self.all_model_params = get_all_params()
        self.geo_emb_cache = geo_emb_cache
        self.geo_emb_calc = None

    def fit(self, geo_embs_batch, wrf_params_batch, targets):
        batch_feats = pd.DataFrame()

        for emb_feats, wrf_params in zip(geo_embs_batch, wrf_params_batch):
            batch_feats = pd.concat([batch_feats, self._merge_feats(emb_feats, wrf_params)])

        for column in self.filtered_columns:
            batch_feats[column] = batch_feats[column].astype('category')

        pool = catboost.Pool(batch_feats,
                             label=targets,
                             cat_features=self.filtered_columns,
                             )

        grid = {
            'learning_rate': [0.007, 0.04, 0.1, 0.4, 1],
            'random_strength': [1, 5, 10, 15, 20],
            'bootstrap_type': ['Bayesian'],
            'l2_leaf_reg': [1, 3, 7, 10],
            'bagging_temperature': [0, 0.25, 0.5, 0.75, 1.0],
            'leaf_estimation_iterations': [1, 3, 7, 10],
            'depth': [4, 6, 10],
            'num_trees': [200, 350, 500],
            'verbose': [50]
        }
        self.regressor = self._create_regressor()
        grid_search_result = self.regressor.grid_search(grid, X=pool)

    def _create_regressor(self):
        return catboost.CatBoostRegressor(loss_function='RMSE', task_type='CPU', train_dir='./catboost')

    def _merge_feats(self, geo_feats, wrf_params):
        res = {}
        if self.geo_scaler is not None:
            geo_feats = self.geo_scaler.transform([geo_feats])[0]
        res.update(self._make_pd_from_geo_emb(geo_feats))
        res.update(self._make_pd_from_wrf_params(wrf_params))
        res = pd.DataFrame(res)
        return res

    def _make_pd_from_geo_emb(self, geo_emb):
        return {
            f"geo_emb_{i}": [feat] for i, feat in enumerate(geo_emb)
        }

    def _make_pd_from_wrf_params(self, wrf_params):
        return {
            key: [value] for key, value in wrf_params.items() if key in self.filtered_columns
        }

    def get_all_predicts(self):
        geo_feats = self.geo_emb_cache
        if self.use_uncertainty:
            preds = [self.regressor.virtual_ensembles_predict(self._merge_feats(geo_feats, model_params), prediction_type='TotalUncertainty',
                                        virtual_ensembles_count=10) for model_params in self.all_model_params]

            preds = np.array(preds)
            assert preds.shape == (len(self.all_model_params), 1, 2)
            predictions = [pred[0][0] for pred in preds]
            uncertanties = [pred[0][1] for pred in preds]
            return copy.deepcopy(self.all_model_params), predictions, uncertanties

        else:
            predictions = [self.regressor.predict(self._merge_feats(geo_feats, model_params)) for model_params in self.all_model_params]
            for pred in predictions:
                assert len(pred) == 1
            predictions = [pred[0] for pred in predictions]
            return copy.deepcopy(self.all_model_params), predictions, None

    def load_model(self, path):
        self.regressor.load_model(path)

    def save_model(self, path):
        self.regressor.save_model(path)

    def set_geo_scaler(self, scaler):
        self.geo_scaler = scaler
