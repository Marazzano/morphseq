import torch
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pytorch_lightning as pl

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, z0, z1, exp_idx, emb_idx, T, dt):
        # store CPU tensors (float32 + int64)
        self.z0, self.z1 = z0, z1
        self.exp_idx, self.emb_idx = exp_idx, emb_idx
        self.dt = dt
        self.temp = T  

    def __len__(self):
        return self.z0.shape[0]

    def __getitem__(self, i):
        dz = (self.z1[i] - self.z0[i]) / self.dt[i]   # finite diff target
        return {
            "z0": self.z0[i],
            "dz": dz,
            "exp": self.exp_idx[i],
            "emb": self.emb_idx[i],
            "temp": self.temp[i] 
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
                        max_dt:int=3600,
                        use_pca:bool=False,
                        n_pca_components:int=10):
    
    """ 
    Builds training data from the embryo DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing embryo metadata.
        min_frames (int): Minimum number of frames for a valid embryo.
        max_dt (int): Maximum time difference in seconds.
        use_pca (bool): Whether to use PCA for dimensionality reduction.
        n_pca_components (int): Number of PCA components to keep if use_pca is True.
    Returns:
        PairDataset: A dataset containing pairs of embeddings and their differences.
    """
    
    # get z col names
    latent_cols = [embryo_df.columns[i] for i in range(1, len(embryo_df.columns)) if embryo_df.columns[i].startswith("z_mu_b")]
    if use_pca:
        # Apply PCA to the mu columns
        pca = PCA(n_components=n_pca_components)
        mu_data = embryo_df[latent_cols].values
        mu_data_pca = pca.fit_transform(mu_data)
        latent_cols = [f"pca_{i}" for i in range(n_pca_components)]
        embryo_df[latent_cols] = mu_data_pca

    # generate IDs
    _, em_id = np.unique(embryo_df["embryo_id"], return_inverse=True)
    _, ex_id = np.unique(embryo_df["experiment_date"].astype(str), return_inverse=True)
    embryo_df["em_id"] = em_id.astype(int)
    embryo_df["ex_id"] = ex_id.astype(int)

    # strip things down
    merge_df0 = embryo_df.loc[:, ["em_id", "ex_id", "experiment_time", "temperature"] + latent_cols]
    counts = merge_df0["em_id"].value_counts()
    merge_df0 = merge_df0[merge_df0["em_id"].map(counts) >= min_frames]
    merge_df0["row_idx"] = merge_df0.groupby("em_id").cumcount() 
    merge_df0 = merge_df0.rename(columns={"experiment_time": "t"})

    # create pairs
    merge_df1 = merge_df0.copy()
    merge_df0["row_idx"] += 1

    merged_df = merge_df0.merge(merge_df1,
                                on=["em_id", "row_idx"],
                                how="inner",
                                suffixes=("_0", "_1")
                            )
    merged_df["dt"] = merged_df["t_1"] - merged_df["t_0"]
    merged_df = merged_df[(merged_df["dt"] > 0) & (merged_df["dt"] <= max_dt)]

    # create data arrays
    z0 = merged_df[[f"{col}_0" for col in latent_cols]].values.astype("float32")
    z1 = merged_df[[f"{col}_1" for col in latent_cols]].values.astype("float32")
    exp_idx = merged_df["ex_id_0"].values
    emb_idx = merged_df["em_id"].values
    dt = merged_df["dt"].values.astype("float32")
    T = merged_df["temperature_0"].values.astype("float32")  # time of first frame

    ds = PairDataset(
                    z0=torch.tensor(z0),
                    z1=torch.tensor(z1),
                    exp_idx=torch.tensor(exp_idx),
                    emb_idx=torch.tensor(emb_idx),
                    dt=torch.tensor(dt), 
                    T=torch.tensor(T) 
                )

    return ds, merged_df.loc[:, ["em_id", "dt"]].reset_index(drop=True)


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


