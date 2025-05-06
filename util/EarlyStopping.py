import torch
from sentence_transformers import SentenceTransformer


class EarlyStopping:
    def __init__(
            self,
            patience=7,
            min_delta=0.,
            name="",
            save_path="",
            save_ckpt=True,
            save_st_model=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False

        self.name = name
        self.save_path = save_path
        self.save_ckpt = save_ckpt
        self.save_st_model = save_st_model

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
            if self.save_ckpt:
                self.save_checkpoint(val_loss, model, self.save_path)
            return

        if val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
            if self.save_ckpt:
                self.save_checkpoint(val_loss, model, self.save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        torch.save(
            obj=model.state_dict(),
            f=f"{save_path}/{self.name}.pt"
        )
        self.best_score = val_loss

    def save_sentence_transformer(self, model_init_path):
        model = SentenceTransformer(model_name_or_path=model_init_path)
        model.load_state_dict(state_dict=self.best_model_state)
        model.save(path=f"{self.save_path}/st_models/{self.name}")
