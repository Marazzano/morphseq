import torch
from torch.utils.data.sampler import SubsetRandomSampler
from src.run.run_utils import initialize_model, parse_model_paths
from torchvision.utils import save_image
from pytorch_lightning import Trainer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import pickle
from src.lightning.pl_wrappers import LitModel
from torch.utils.data import DataLoader
from src.data.dataset_configs import BaseDataConfig
from src.analyze.assess_hydra_results import get_hydra_runs, initialize_model_to_asses, parse_hydra_paths
import shutil

torch.set_float32_matmul_precision("medium")   # good default


def copy_config_to_outfolder(cfg_path, out_folder):
    """
    Copies the Hydra config file to the output folder for provenance.
    If the config is a dictionary (in-memory), saves it as YAML.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    if isinstance(cfg_path, str) and os.path.isfile(cfg_path):
        # Config is a file on disk
        shutil.copy2(cfg_path, out_folder / "config.yaml")
    else:
        # Config is a dict or OmegaConf object
        import yaml
        config_path = out_folder / "config.yaml"
        with open(config_path, "w") as f:
            if hasattr(cfg_path, "to_container"):
                # OmegaConf object
                yaml.safe_dump(cfg_path.to_container(resolve=True), f)
            else:
                yaml.safe_dump(cfg_path, f)


def recon_wrapper(
    hydra_run_path,
    run_type,
    out_path,
    n_image_figures=64,
    random_seed=42,        # <-- ensure deterministic sampling
):
    _, cfg_path_list = get_hydra_runs(hydra_run_path, run_type)

    for cfg in tqdm(cfg_path_list, "Processing training runs..."):
        if isinstance(cfg, str):
            config = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            config = cfg
        else:
            raise Exception("cfg argument dtype is not recognized")

        # initialize
        try:
            model, model_config = initialize_model_to_asses(config)
        except Exception:
            continue
        loss_fn = model_config.lossconfig.create_module()
        run_path = os.path.dirname(os.path.dirname(cfg))
        latest_ckpt = parse_hydra_paths(run_path=run_path)
        if latest_ckpt is None:
            continue

        # load train/test/eval indices
        split_path = os.path.join(run_path, "split_indices.pkl")
        with open(split_path, 'rb') as file:
            split_dict = pickle.load(file)

        # initialize new data config for evaluation
        eval_data_config = BaseDataConfig(
            train_indices=np.asarray(split_dict["train"]),
            test_indices=np.asarray(split_dict["test"]),
            eval_indices=np.asarray(split_dict["eval"]),
            root=os.path.dirname(os.path.dirname(os.path.dirname(run_path))),
            return_sample_names=True,
            transform_name="basic"
        )
        eval_data_config.make_metadata()

        # load model
        lit_model = LitModel.load_from_checkpoint(
            latest_ckpt,
            model=model,
            loss_fn=loss_fn,
            data_cfg=eval_data_config
        )

        lit_model.eval()
        lit_model.freeze()

        print("Evaluating model " + os.path.basename(run_path) + ')')

        dataset = eval_data_config.create_dataset()

        # deterministic random selection
        load_indices = getattr(eval_data_config, "test_indices")
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        shuffled_indices = np.random.permutation(load_indices)
        selected_indices = shuffled_indices[:n_image_figures]
        sampler = SubsetRandomSampler(selected_indices)

        dl = DataLoader(
            dataset,
            batch_size=n_image_figures,
            num_workers=eval_data_config.num_workers,
            sampler=sampler,
            shuffle=False,
        )

        # construct out path
        # folder_name = os.path.basename(os.path.dirname(run_path))
        # mdl_name = model_config.ddconfig.name
        # latent_dim = model_config.ddconfig.latent_dim
        # pips_wt = model_config.lossconfig.pips_weight
        # gan_wt = model_config.lossconfig.gan_weight
        # pixel_loss = model_config.lossconfig.reconstruction_loss
        # if gan_wt == 0:
        #     gan_str = "noGAN"
        # else:
        #     gan_str = f"GAN_{model_config.lossconfig.gan_net}"
        # attn = model_config.ddconfig.dec_use_local_attn
        # f_stub = folder_name.split("_")[0]
        # out_name = (
        #     f"{mdl_name}_z{int(latent_dim):03}_p{int(10*pips_wt)}_g{int(np.ceil(100*gan_wt))}_attn_{attn}_" + \
        #     gan_str + f"_{f_stub}_" + f"{pixel_loss}_" + Path(run_path).name
        # )
        out_name = Path(run_path).parent.name + "_" + Path(run_path).name
        mdl_folder = os.path.join(out_path, out_name)
        os.makedirs(mdl_folder, exist_ok=True)
        copy_config_to_outfolder(cfg, mdl_folder)

        assess_image_reconstructions(
            lit_model=lit_model,
            dataloader=dl,
            out_dir=mdl_folder,
            device=lit_model.device
        )


def assess_image_reconstructions(
    lit_model: LitModel,
    dataloader: torch.utils.data.DataLoader,
    out_dir: str,
    device: str | torch.device = "cuda",
):
    lit_model.to(device).eval().freeze()
    trainer = Trainer(accelerator="auto", devices=1, limit_predict_batches=1)
    lit_model.current_mode = "test"
    preds = trainer.predict(lit_model, dataloaders=dataloader)

    for p in preds:
        for i in range(p["orig"].size(0)):
            snip_name = os.path.basename(p['snip_ids'][i]).replace(".jpg", "")
            fpath = os.path.join(
                out_dir, f"{snip_name}_loss{int(p['recon_loss'][i]):05}.png"
            )
            grid = torch.stack([p["orig"][i], p["recon"][i]], dim=0)
            save_image(grid, fpath, nrow=2, pad_value=1)
