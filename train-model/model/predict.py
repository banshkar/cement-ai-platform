import numpy as np
import xgboost as xgb

class Model:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("model.bst")

    def predict(self, instances):
        """
        Args:
            instances: list of feature rows. Vertex may send a single row as 1D list.
        Returns:
            list of predictions
        """
        X = np.array(instances, dtype=np.float32)

        # ✅ Fix: Ensure 2D shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dmatrix = xgb.DMatrix(X)
        preds = self.model.predict(dmatrix)
        return preds.tolist()

