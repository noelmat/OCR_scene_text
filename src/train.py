import dataset
import config
import datautils


def run():
    path = datautils.Path('../input/train_data')
    df = datautils.create_label_df(path)
    enc = datautils.create_encoding(df)

    print(enc.classes_)


if __name__ == "__main__":
    run()
