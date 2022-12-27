import importlib


def get_model(request, optimizer_dict, model_conf_dict, train_dataloader, val_dataloader, device):
     # model exists - has been validated by the supervisor
    model_module = importlib.import_module("modyn.models." + request.model_id)

    # this is tailor-made for a specific example
    # TODO(fotstrt): generalize
    model = model_module.Model(
        request.torch_optimizer,
        optimizer_dict,
        model_conf_dict,
        train_dataloader,
        val_dataloader,
        device
    )

    return model