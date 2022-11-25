from pathlib import Path
import torch
import torch.nn.functional as F


class Callback:
    def on_optimizer_step(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_model_load(self, *args, **kwargs):
        pass

    def on_model_save(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass


def filter_requires_grad_parameters(parameters):
    return [v for v in parameters if v.requires_grad]


def filter_requires_grad(model_dict):
    return {k: v for k, v in model_dict.items() if v.requires_grad}


def state_dict_filtered_reqgrad(model):
    state_dict = model.state_dict()
    exclude_param_names = set(
        [k for k, v in model.named_parameters() if not v.requires_grad]
    )
    exclude_param_names = [
        ".".join(k.split(".")[2:]) for k in exclude_param_names
    ]  # Workaround for pytorch lightning; should be TESTED with different distributed training schemes (other than ddp)
    to_return = {k: v for k, v in state_dict.items() if k not in exclude_param_names}
    return to_return


class ModelSaver:
    def __init__(self):
        pass

    def load(self, args, train_module, model, optimizer, discriminator, optimizer_d):
        if args.resume is not None:
            checkpoint = train_module.load(args.resume)
            model.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            discriminator.load_state_dict(checkpoint["discriminator"], strict=False)
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            args.start_epoch = checkpoint["epoch"] + 1

    def save(
        self, args, epoch, train_module, model, optimizer, discriminator, optimizer_d
    ):
        if train_module.is_global_zero:
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / f"checkpoint-{epoch}.pth"]
            to_save = {
                "args": args,
                "epoch": epoch,
                "model": state_dict_filtered_reqgrad(model)
                if epoch > 0
                else model.state_dict(),
                # "model": state_dict_filtered_reqgrad(model),
                "optimizer": optimizer.state_dict(),
                "discriminator": state_dict_filtered_reqgrad(discriminator),
                "optimizer_d": optimizer_d.state_dict(),
            }
            for checkpoint_path in checkpoint_paths:
                train_module.save(to_save, checkpoint_path)
            train_module.save(
                dict(args=to_save["args"]), output_dir / f"checkpoint-{epoch}.args"
            )


def get_imgs_to_log(dataset, n_imgs):
    return next(
        iter(
            torch.utils.data.DataLoader(
                dataset, batch_size=n_imgs, shuffle=True, num_workers=2
            )
        )
    )[0]
