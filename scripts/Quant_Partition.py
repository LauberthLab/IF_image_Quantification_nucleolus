import os
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io as skio, measure
from skimage.color import rgb2gray

try:
    from .utils import _ensure_dir, parent_by_mode
except ImportError:
    from utils import _ensure_dir, parent_by_mode


def _load_label_image(path):
    arr = np.squeeze(skio.imread(path))
    if arr.ndim == 3:
        arr = (rgb2gray(arr) > 0).astype(np.uint8)
    if arr.dtype.kind in "fc" or arr.max() <= 1:
        arr = (arr > 0).astype(np.uint8)
    return measure.label(arr > 0).astype(np.int32)


def _load_intensity_image(path):
    arr = np.squeeze(skio.imread(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def analyze_partition_and_radial_distribution(
    nd2_file,
    nuclei_labels_path,
    nucleoli_labels_path,
    foci_labels_path,
    target_intensity_path,
    target_name="UBF",
    output_dir="./partition_results",
    group_name="",
):
    """
    Compute two complementary metrics using outputs from Colocalization.py:

    1) Partition coefficient (per nucleolus):
       log2(mean intensity in nucleolus / mean intensity in nucleoplasm)
       where nucleoplasm = nucleus - all nucleoli in that nucleus.

    2) Focus radial metric (per focus):
       distance-to-rim and distance-to-centroid for each focus centroid,
       then a normalized outward score in [0,1]:
       dist_to_centroid / (dist_to_centroid + dist_to_rim)

    A score near 1 means focus is close to rim; near 0 means near center.
    """

    file_stem = os.path.splitext(os.path.basename(nd2_file))[0]
    prefix = f"{group_name}_" if group_name else ""
    out_root = _ensure_dir(os.path.join(output_dir, prefix + file_stem))

    nuclei = _load_label_image(nuclei_labels_path)
    nucleoli = _load_label_image(nucleoli_labels_path)
    foci = _load_label_image(foci_labels_path)
    target_img = _load_intensity_image(target_intensity_path)

    if nuclei.shape != nucleoli.shape or nuclei.shape != foci.shape or nuclei.shape != target_img.shape:
        raise ValueError("All inputs must have matching XY dimensions.")

    # Constrain labels to nuclei footprint.
    nucleoli = np.where(nuclei > 0, nucleoli, 0)
    nucleoli = measure.label(nucleoli > 0).astype(np.int32)
    foci = np.where(nuclei > 0, foci, 0)
    foci = measure.label(foci > 0).astype(np.int32)

    nuc_of_nucleolus = parent_by_mode(nucleoli, nuclei)
    nucleolus_of_focus = parent_by_mode(foci, nucleoli)

    nucleoli_props = {p.label: p for p in measure.regionprops(nucleoli)}

    # ----- per-nucleolus partition coefficient -----
    rows_nucleolus = []
    for nl, nuc_id in nuc_of_nucleolus.items():
        if nuc_id == 0:
            continue

        nucleolus_mask = nucleoli == nl
        nucleus_mask = nuclei == nuc_id
        nucleoplasm_mask = nucleus_mask & (nucleoli == 0)

        mean_nucleolus = float(target_img[nucleolus_mask].mean()) if nucleolus_mask.any() else np.nan
        mean_nucleoplasm = float(target_img[nucleoplasm_mask].mean()) if nucleoplasm_mask.any() else np.nan

        if np.isfinite(mean_nucleolus) and np.isfinite(mean_nucleoplasm):
            partition_log2 = float(np.log2((mean_nucleolus + 1e-9) / (mean_nucleoplasm + 1e-9)))
        else:
            partition_log2 = np.nan

        rows_nucleolus.append(
            {
                "nucleus_label": int(nuc_id),
                "nucleolus_label": int(nl),
                f"mean_intensity_{target_name}_nucleolus": mean_nucleolus,
                f"mean_intensity_{target_name}_nucleoplasm": mean_nucleoplasm,
                f"partition_coeff_log2_{target_name}": partition_log2,
            }
        )

    df_nucleolus = pd.DataFrame(rows_nucleolus)

    # ----- per-focus radial movement metric -----
    rows_focus = []
    for fp in measure.regionprops(foci, intensity_image=target_img):
        f_label = fp.label
        n_label = int(nucleolus_of_focus.get(f_label, 0))
        if n_label == 0 or n_label not in nucleoli_props:
            continue

        nprop = nucleoli_props[n_label]
        minr, minc, maxr, maxc = nprop.bbox

        local_nucleolus = (nucleoli[minr:maxr, minc:maxc] == n_label)
        dist_to_rim_map = ndi.distance_transform_edt(local_nucleolus)

        cy, cx = fp.centroid
        y = int(np.clip(round(cy) - minr, 0, dist_to_rim_map.shape[0] - 1))
        x = int(np.clip(round(cx) - minc, 0, dist_to_rim_map.shape[1] - 1))

        dist_to_rim = float(dist_to_rim_map[y, x])
        ncy, ncx = nprop.centroid
        dist_to_centroid = float(np.hypot(cy - ncy, cx - ncx))

        outward_score = dist_to_centroid / (dist_to_centroid + dist_to_rim + 1e-9)
        rim_to_center_ratio = dist_to_rim / (dist_to_centroid + 1e-9)

        rows_focus.append(
            {
                "focus_label": int(f_label),
                "nucleolus_label": int(n_label),
                "nucleus_label": int(nuc_of_nucleolus.get(n_label, 0)),
                "focus_area_px": int(fp.area),
                f"focus_mean_intensity_{target_name}": float(fp.mean_intensity),
                "focus_centroid_y": float(cy),
                "focus_centroid_x": float(cx),
                "dist_to_rim_px": dist_to_rim,
                "dist_to_nucleolus_centroid_px": dist_to_centroid,
                "outward_score_0_center_1_rim": float(outward_score),
                "rim_to_center_distance_ratio": float(rim_to_center_ratio),
            }
        )

    df_focus = pd.DataFrame(rows_focus)

    # ----- aggregate per nucleus -----
    if not df_nucleolus.empty:
        nuc_partition = (
            df_nucleolus.groupby("nucleus_label")[f"partition_coeff_log2_{target_name}"]
            .mean()
            .reset_index()
            .rename(columns={f"partition_coeff_log2_{target_name}": f"mean_partition_coeff_log2_{target_name}"})
        )
    else:
        nuc_partition = pd.DataFrame(columns=["nucleus_label", f"mean_partition_coeff_log2_{target_name}"])

    if not df_focus.empty:
        nuc_radial = (
            df_focus.groupby("nucleus_label")
            .agg(
                mean_outward_score=("outward_score_0_center_1_rim", "mean"),
                median_outward_score=("outward_score_0_center_1_rim", "median"),
                mean_dist_to_rim_px=("dist_to_rim_px", "mean"),
                n_foci=("focus_label", "count"),
            )
            .reset_index()
        )
    else:
        nuc_radial = pd.DataFrame(
            columns=["nucleus_label", "mean_outward_score", "median_outward_score", "mean_dist_to_rim_px", "n_foci"]
        )

    df_nucleus = nuc_partition.merge(nuc_radial, on="nucleus_label", how="outer")

    nucleolus_csv = os.path.join(out_root, f"per_nucleolus_partition_{target_name}.csv")
    focus_csv = os.path.join(out_root, f"per_focus_radial_{target_name}.csv")
    nucleus_csv = os.path.join(out_root, f"per_nucleus_partition_radial_{target_name}.csv")

    df_nucleolus.to_csv(nucleolus_csv, index=False)
    df_focus.to_csv(focus_csv, index=False)
    df_nucleus.to_csv(nucleus_csv, index=False)

    return {
        "out_dir": out_root,
        "per_nucleolus_csv": nucleolus_csv,
        "per_focus_csv": focus_csv,
        "per_nucleus_csv": nucleus_csv,
        "n_nucleoli": int(len(df_nucleolus)),
        "n_foci": int(len(df_focus)),
    }


def _build_parser():
    p = argparse.ArgumentParser(description="Partition coefficient + radial foci analysis from Colocalization outputs.")
    p.add_argument("--nd2_file", required=True)
    p.add_argument("--nuclei_labels_path", required=True)
    p.add_argument("--nucleoli_labels_path", required=True)
    p.add_argument("--foci_labels_path", required=True, help="Use nc_labels.tif from Colocalization output")
    p.add_argument("--target_intensity_path", required=True, help="Use raw_<target>.png or tif from Colocalization output")
    p.add_argument("--target_name", default="UBF")
    p.add_argument("--output_dir", default="./partition_results")
    p.add_argument("--group_name", default="")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    res = analyze_partition_and_radial_distribution(
        nd2_file=args.nd2_file,
        nuclei_labels_path=args.nuclei_labels_path,
        nucleoli_labels_path=args.nucleoli_labels_path,
        foci_labels_path=args.foci_labels_path,
        target_intensity_path=args.target_intensity_path,
        target_name=args.target_name,
        output_dir=args.output_dir,
        group_name=args.group_name,
    )
    print(res)
