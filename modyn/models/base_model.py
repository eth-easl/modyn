import torch

class BaseModel():
    def __init__(
        self,
        train_loader,
        val_loader,
        device,
        checkpoint_path,
        checkpoint_interval,
    ):

        self._model = None
        self._optimizer = None

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval = checkpoint_interval


    def save_checkpoint(self, iteration):

        # TODO(fotstrt): this might overwrite checkpoints from previous runs
        # we could have a counter for the specific training, and increment it
        # every time a new checkpoint is saved.

        # TODO: we assume a local checkpoint for now,
        # should we add functionality for remote?
        checkpoint_file_name = self._checkpoint_path + f'/model_{iteration}' + '.pt'
        dict_to_save = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        torch.save(dict_to_save, checkpoint_file_name)


    def load_checkpoint(self, path):

        checkpoint_dict = torch.load(path)
        assert 'model' in checkpoint_dict
        assert 'optimizer' in checkpoint_dict
        self._model.load_state_dict(checkpoint_dict['model'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])
