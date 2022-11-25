import math
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils_move import AvgLogDict


def train_one_epoch(
    args,
    data_loader,
    model,
    discriminator,
    optimizer,
    optimizer_d,
    epoch,
    modifiers_gen,
    modifiers_disc,
    losses,
    train_module,
    gen_keys=["composed"],
    fake_keys=["composed", "real_ae_shift_cp"],
    real_keys=["composed_noshift", "real_ae"],
    accum_iter=1,
    log_every=10,
):
    assert log_every > 1  # here one iteration is either segmenter or discriminator
    model.train(True)
    discriminator.train(True)

    dl_len = len(data_loader)

    phases = [
        "discriminator",
        "segmenter",
    ]  # this is the order in which we train the models
    cur_phase_id = 1

    log_dict = AvgLogDict()

    pbar = tqdm(enumerate(data_loader), total=dl_len)

    for data_iter_step, (samples, _) in pbar:
        global_iter = epoch * dl_len + data_iter_step
        if data_iter_step % accum_iter == 0:
            cur_phase_id = (cur_phase_id + 1) % len(phases)

        cur_phase = phases[cur_phase_id]

        if cur_phase == "discriminator":
            with torch.no_grad():
                return_dict = model(samples)
                return_dict = modifiers_disc(
                    return_dict
                )  # apply modifiers to the output; should be a ComposeModifier
            if "real" in real_keys:
                return_dict["real"] = samples

            # Get the fake and real inputs for the discriminator
            fake_inputs = [return_dict[k] for k in fake_keys]
            real_inputs = [return_dict[k] for k in real_keys]

            # Run the discriminator
            fake_disc_outputs = {
                k: discriminator.run_D(fake_input)
                for k, fake_input in zip(fake_keys, fake_inputs)
            }
            real_disc_outputs = {
                k: discriminator.run_D(real_input)
                for k, real_input in zip(real_keys, real_inputs)
            }

            # Get the loss for the discriminator
            disc_losses, disc_log_dict = discriminator.loss_d(
                fake_disc_outputs, real_disc_outputs
            )
            loss = sum(disc_losses.values())

            disc_losses["loss_d"] = loss

            if global_iter // len(phases) % log_every == 0:
                [
                    log_dict.__setitem__(k, v.detach().cpu().item())
                    for k, v in disc_losses.items()
                ]
                [
                    log_dict.__setitem__(
                        k, v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    )
                    for k, v in disc_log_dict.items()
                ]

        elif cur_phase == "segmenter":
            return_dict = model(samples)
            return_dict = modifiers_gen(
                return_dict
            )  # apply modifiers to the output; should be a ComposeModifier

            # Compute mask-related losses
            scaled_loss_dict, loss_dict = losses(return_dict)

            # Adversarial loss
            gen_disc_inputs = [return_dict[k] for k in gen_keys]
            gen_disc_outputs = {
                k: discriminator.run_D(gen_disc_input)
                for k, gen_disc_input in zip(gen_keys, gen_disc_inputs)
            }
            gen_losses, gen_log_dict = discriminator.loss_g(gen_disc_outputs)

            loss_g = sum(gen_losses.values())
            gen_losses["loss_g"] = loss_g
            loss_dict.update(gen_losses)
            loss = sum(scaled_loss_dict.values()) + loss_g
            loss_dict["loss_seg"] = loss

            if global_iter // len(phases) % log_every == 0:
                [
                    log_dict.__setitem__(k, v.detach().cpu().item())
                    for k, v in loss_dict.items()
                ]
                [
                    log_dict.__setitem__(k, v.detach().cpu().item())
                    for k, v in gen_log_dict.items()
                ]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        train_module.backward(loss)
        if (data_iter_step + 1) % accum_iter == 0:
            if cur_phase == "segmenter":
                optimizer.step()
            elif cur_phase == "discriminator":
                optimizer_d.step()

            optimizer.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

        if train_module.is_global_zero:
            # Use your favorite logger to log losses from log_dict
            if (
                (global_iter // len(phases) % log_every == 0)
                and ("loss_g" in log_dict.dict_)
                and ("loss_d" in log_dict.dict_)
            ):
                pbar.set_description(
                    f'Iter: {global_iter}, LossD: {log_dict["loss_d"]:.4f}, LossG: {log_dict["loss_g"]:.4f}'
                )
                log_dict = AvgLogDict()
