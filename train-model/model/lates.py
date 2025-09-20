import xgboost as xgb
booster = xgb.Booster()
booster.load_model("model.bst")
print(booster.feature_names)
