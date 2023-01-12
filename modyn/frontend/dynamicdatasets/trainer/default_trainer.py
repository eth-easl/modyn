import copy
import logging
import sys
import time

import torch
import Typing
import yaml
from tqdm import tqdm
from trainer import Trainer

logging.basicConfig(format="%(asctime)s %(message)s")


class DefaultTrainer(Trainer):
    def __init__(self, config: dict):
        super().__init__(config)

    def _train(self) -> Typing.any:
        logging.info("Training with Default Trainer")
        since = time.time()

        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0

        for epoch in range(self._num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, self._num_epochs))
            logging.info("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self._model.train()  # Set model to training mode
                else:
                    self._model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(self._dataloaders[phase]):
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                    # zero the parameter gradients
                    self._optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)

                        loss = self._criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self._optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    self._scheduler.step()

                epoch_loss = running_loss / len(self._dataloaders[phase].dataset)
                epoch_acc = float(running_corrects) / len(
                    self._dataloaders[phase].dataset
                )

                logging.info(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self._model.state_dict())

        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        logging.info("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        self._model.load_state_dict(best_model_wts)

        return self._model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python trainer.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    dt = DefaultTrainer(config)
    dt.train()
