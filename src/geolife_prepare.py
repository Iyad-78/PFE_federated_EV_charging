import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def _read_one_plt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "unused", "alt_ft", "days", "date", "time"],
        low_memory=True
    )

    df["dt"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["dt", "lat", "lon"])

    # IMPORTANT: filter invalid GPS ranges
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]

    df = df[["dt", "lat", "lon", "alt_ft"]]
    return df


def prepare_geolife_points(geolife_root: str, out_parquet: str,
                           max_users: int = 60,
                           max_trajectories_per_user: int = 40,
                           max_points_total: int = 2_000_000) -> dict:
    """
    Parcourt geolife_root/Data/000..181/Trajectory/*.plt et écrit un parquet de points.
    Structure dataset : Data/ contient des dossiers user '000' à '181', chacun avec Trajectory/*.plt. :contentReference[oaicite:4]{index=4}
    """
    root = Path(geolife_root)
    data_dir = root / "Data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected {data_dir} (GeoLife root must contain Data/)")

    user_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    user_dirs = user_dirs[:max_users]

    rows = []
    n_points = 0

    for user_dir in tqdm(user_dirs, desc="Users"):
        uid = user_dir.name  # '000', '001', ...
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue

        plt_files = sorted(traj_dir.glob("*.plt"))[:max_trajectories_per_user]
        for f in plt_files:
            df = _read_one_plt(f)
            if df.empty:
                continue
            df["user_id"] = uid
            df["traj_id"] = f.stem  # nom du fichier = timestamp de départ
            rows.append(df)
            n_points += len(df)

            if n_points >= max_points_total:
                break
        if n_points >= max_points_total:
            break

    if not rows:
        raise RuntimeError("No points parsed. Check dataset path / format.")

    out = pd.concat(rows, ignore_index=True)
    # conversion altitude feet -> meters (optionnel, mais propre)
    out["alt_m"] = out["alt_ft"].astype(float) * 0.3048
    out = out.drop(columns=["alt_ft"])

    os.makedirs(Path(out_parquet).parent, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    summary = {
        "users_parsed": int(out["user_id"].nunique()),
        "trajectories_parsed": int(out[["user_id", "traj_id"]].drop_duplicates().shape[0]),
        "points": int(len(out)),
    }
    print("GeoLife points saved:", summary)
    return summary
