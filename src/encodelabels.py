import datautils
from datautils import Path
import joblib


def encode_labels(path=None):
    path = Path('../input/train_data') if path is None else Path(path)
    df = datautils.create_label_df(path)
    encoder = datautils.create_encoding(df)
    print(encoder.classes_)
    return encoder


if __name__ == "__main__":
    encoder = encode_labels()
    joblib.dump(encoder, 'label_encoder.pkl',protocol=2)
