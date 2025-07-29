import torch
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pytorch_lightning as pl

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, z0, z1, exp_idx, emb_idx, dt, temp):
        self.z0 = z0
        self.z1 = z1
        self.exp_idx = exp_idx
        self.emb_idx = emb_idx
        self.dt = dt
        self.temp = temp

    def __len__(self):
        return self.z0.shape[0]

    def __getitem__(self, i):
        return {
            "z0": self.z0[i],
            "z1": self.z1[i],
            "dt": self.dt[i] / 3600.0,
            "exp": self.exp_idx[i],
            "emb": self.emb_idx[i],
            "temp": self.temp[i],
        }
    


def load_embryo_df(root:Path, 
                   model_class: str= "legacy", 
                   model_name: str="20241107_ds_sweep01_optimum"):
    

    data_path = root / "models" / model_class / model_name
    if not data_path.exists():
        raise FileNotFoundError(f"Model path {data_path} does not exist.")
    
    # Load the embeddings
    embryo_df = pd.read_csv(data_path / "embryo_stats_df.csv", index_col=0)
    if embryo_df.columns[0].startswith("Unnamed"):
        embryo_df.set_index(embryo_df.columns[0], inplace=True)

    return embryo_df


def get_data_splits(df: pd.DataFrame, 
                    train_fraction: float = 0.8, 
                    val_fraction: float = 0.1):
    """
    Splits the embryo DataFrame into training, validation, and test sets.
    
    Args:
        embryo_df (pd.DataFrame): DataFrame containing embryo metadata.
        train_fraction (float): Fraction of data to use for training.
        val_fraction (float): Fraction of data to use for validation.
        
    Returns:
        tuple: Three DataFrames for training, validation, and test sets.
    """
    
    df.reset_index(inplace=True)  # ensure 'embryo_id' is a column
    
    # Suppose your df is already loaded and has 'embryo_id'\
    test_fraction = 1-train_fraction-val_fraction
    desired_props = dict(train=train_fraction, val=val_fraction, test=test_fraction)

    # Count rows per embryo
    counts = df['em_id'].value_counts().sort_values(ascending=False)
    embryos = counts.index.to_numpy()
    sizes   = counts.values

    # Shuffle for randomness
    rng = np.random.default_rng(42)
    shuf = rng.permutation(len(embryos))
    embryos, sizes = embryos[shuf], sizes[shuf]

    splits = {}
    cum = 0
    n_rows = len(df)
    for split, prop in desired_props.items():
        n_split = int(round(n_rows * prop))
        splits[split] = []
        split_size = 0
        while cum < n_rows and split_size < n_split and len(embryos) > 0:
            eid, esize = embryos[0], sizes[0]
            splits[split].append(eid)
            split_size += esize
            cum += esize
            embryos, sizes = embryos[1:], sizes[1:]

    # If any embryos are left, add them to the last split
    if len(embryos) > 0:
        splits[split].extend(embryos.tolist())

    # Now get row indices for each split
    split_indices = {
        split: df[df['em_id'].isin(eids)].index.to_numpy()
        for split, eids in splits.items()
    }
    
    return split_indices


def build_training_data(embryo_df: pd.DataFrame, 
                        min_frames:int=5,
                        max_dt:int=36000,
                        use_pca:bool=False,
                        n_pca_components:int=10,
                        n_steps:int=3):
    """Build training pairs with fixed temporal step size (n_steps)."""
    latent_cols = [c for c in embryo_df.columns if c.startswith("z_mu_b")]
    if use_pca:
        pca = PCA(n_components=n_pca_components)
        pca_vals = pca.fit_transform(embryo_df[latent_cols].values)
        latent_cols = [f"pca_{i}" for i in range(n_pca_components)]
        embryo_df[latent_cols] = pca_vals

    _, em_id = np.unique(embryo_df["embryo_id"], return_inverse=True)
    _, ex_id = np.unique(embryo_df["experiment_date"].astype(str), return_inverse=True)
    embryo_df["em_id"] = em_id.astype(int)
    embryo_df["ex_id"] = ex_id.astype(int)

    # Prune short embryos
    embryo_df["row_idx"] = embryo_df.groupby("em_id").cumcount()
    counts = embryo_df["em_id"].value_counts()
    embryo_df = embryo_df[embryo_df["em_id"].map(counts) >= min_frames]

    df0 = embryo_df.copy()
    df1 = embryo_df.copy()
    df1["row_idx"] -= n_steps

    merged = df0.merge(df1, on=["em_id", "row_idx"], suffixes=("_0", "_1"))
    merged["dt"] = merged["experiment_time_1"] - merged["experiment_time_0"]
    merged = merged[(merged["dt"] > 0) & (merged["dt"] <= max_dt)]

    z0 = merged[[f"{c}_1" for c in latent_cols]].values.astype("float32")
    z1 = merged[[f"{c}_0" for c in latent_cols]].values.astype("float32")
    dt = merged["dt"].values.astype("float32")
    temp = merged["temperature_1"].values.astype("float32")
    exp_idx = merged["ex_id_1"].values.astype("int64")
    emb_idx = merged["em_id"].values.astype("int64")

    # generate stripped-down dataset
    # where we only keep the necessary columns
    df_out = merged[["ex_id_1", "em_id", "snip_id_0", "embryo_id_0", "experiment_date_0", "experiment_time_1", "dt"] + [f"{c}_0" for c in latent_cols]].copy()
    df_out.columns = ["_".join(col.split("_")[:-1]) for col in df_out.columns]

    ds = PairDataset(torch.tensor(z0), torch.tensor(z1), 
                     torch.tensor(exp_idx), torch.tensor(emb_idx), 
                     torch.tensor(dt), torch.tensor(temp))
    return ds, merged[["em_id", "dt"]].reset_index(drop=True), df_out


# class NVFData(pl.LightningDataModule):
#     def __init__(self, *arrays, batch_size=8192):
#         super().__init__()
#         self.arrays, self.bs = arrays, batch_size
#     def setup(self, stage=None):
#         self.ds = PairDataset(*self.arrays)
#     def train_dataloader(self):
#         return DataLoader(self.ds, batch_size=self.bs, shuffle=True, pin_memory=True)

if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
    model_class = "legacy"
    model_name = "20241107_ds_sweep01_optimum"

    embryo_df = load_embryo_df(root, model_class, model_name)
    test = build_training_data(embryo_df=embryo_df,)
    test[0]
    print("check")


