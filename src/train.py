import dataset
import config
import datautils
import model
import engine
from sklearn.model_selection import train_test_split
import joblib
import torch



def run():
    path = datautils.Path('../input/train_data')
    image_files = datautils.get_images(path)
    train_paths, valid_paths = train_test_split(image_files, test_size=config.VALID_SPLIT, random_state=42)
    print(len(train_paths),len(valid_paths))
    encoder = joblib.load('label_encoder.pkl')
    train_ds = dataset.Dataset(train_paths,get_labels=datautils.get_label, label_enc=encoder, size=(1200,600))
    num_classes = len(encoder.classes_)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=8,
        shuffle=True
    )
    
    valid_ds = dataset.Dataset(valid_paths,get_labels=datautils.get_label, label_enc=encoder, size=(1200,600))

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=config.TRAIN_BATCH_SIZE*2,
        num_workers=8,
        shuffle=False
    )
    
    ocr_model = model.Model(len(encoder.classes_))
    ocr_model.to(config.DEVICE)

    total_steps = len(train_dl) * config.N_EPOCHS

    opt = torch.optim.Adam(ocr_model.parameters(), config.MAX_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        config.MAX_LR,
        total_steps=total_steps)
    
    for epoch in range(config.N_EPOCHS):
        engine.train_loop(train_dl,ocr_model, opt, scheduler, None, config.DEVICE)
        losses, output = engine.eval_loop(valid_dl, ocr_model, None, config.DEVICE)

        print(torch.tensor(losses).mean().item())

    save_dict = {
        'label_encoding': encoder,
        'model_dict': ocr_model.state_dict()
    }
    torch.save(save_dict, f'ocr_model_{config.N_EPOCHS}')

if __name__ == "__main__":
    run()
