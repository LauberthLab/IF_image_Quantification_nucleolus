import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nd2
from csbdeep.utils import normalize

import skimage.filters as filters
import skimage.morphology as morphology
import skimage.measure as measure
import skimage.color as color
import skimage.io as skio
from skimage.color import rgb2gray
# --------------------------------------------------------------------------------------------

try:
    from .utils import (
        parent_by_mode, normalize01, _ensure_dir, _save_gray, _save_label_tiff,
        overlay_labels, flatfield_correct
    )
except ImportError:
    from utils import (
        parent_by_mode, normalize01, _ensure_dir, _save_gray, _save_label_tiff,
        overlay_labels, flatfield_correct
    )

def get_CoLoc(
    nd2_file,
    nuclei_labels_path,
    npm1_labels_path,
    DAPI_IDX=0,
    NPM1_IDX=2,
    NC_IDX=3,
    DAPI_NAME="DAPI",
    NPM1_NAME="NPM1",
    nucleolar_component = None,
    PIXEL_SIZE_UM=None,
    output_dir="./",
    group_name="",
    # saving
    save_png=True, save_tiff=True, save_numpy=False, save_figure=True, figure_dpi=600,
):
    """
    Reuse precomputed nuclei + NPM1 (nucleoli) labels from StarDist,
    segment FBL foci, and compute Manders M1/M2 between NPM1 and FBL.
    """

    file_stem = os.path.splitext(os.path.basename(nd2_file))[0]
    prefix = f"{group_name}_" if group_name else ""
    out_root = _ensure_dir(os.path.join(output_dir, prefix + file_stem))

        
    rdr = nd2.ND2File(nd2_file)
    arr_raw = rdr.asarray()

    # reshape to (C, Y, X) by max projecting Z/T if present
    if arr_raw.ndim == 5:      # (T,Z,C,Y,X)
        T, Z, C, Y, X = arr_raw.shape
        arr = arr_raw.max(axis=(0,1))       # -> (C,Y,X)
        chan_count = C
        chan_names = []
        try:
            md = rdr.metadata
            if hasattr(md, "channels") and md.channels:
                chan_names = [getattr(ch, "name", f"ch{i}") or f"ch{i}"
                              for i, ch in enumerate(md.channels)]
        except Exception:
            pass
    elif arr_raw.ndim == 4:    # (Z,C,Y,X) or (T,C,Y,X)
        arr = arr_raw.max(axis=0)           # -> (C,Y,X)
        chan_count = arr.shape[0]
        chan_names = [f"ch{i}" for i in range(chan_count)]
    elif arr_raw.ndim == 3:    # (C,Y,X)
        arr = arr_raw
        chan_count = arr.shape[0]
        chan_names = [f"ch{i}" for i in range(chan_count)]
    elif arr_raw.ndim == 2:    # (Y,X)
        arr = arr_raw[None, ...]; chan_count = 1; chan_names = ["ch0"]
    else:
        rdr.close()
        raise ValueError(f"Unexpected ND2 shape: {arr_raw.shape}")
    rdr.close()

    def _resolve_channel(name_hint, fallback_idx, names, nC):
        if isinstance(fallback_idx, (list, tuple, np.ndarray)):
            fallback_idx = int(fallback_idx[0])
        if names:
            lname = [str(s).lower() for s in names]
            target = (str(name_hint).lower() if name_hint is not None else "")
            if target:
                for i, s in enumerate(lname):
                    if s == target:
                        return int(i)
                for i, s in enumerate(lname):
                    if target in s:
                        return int(i)
        if isinstance(fallback_idx, (int, np.integer)) and 0 <= fallback_idx < nC:
            return int(fallback_idx)
        return 0

    dapi_ch = int(np.clip(_resolve_channel(DAPI_NAME, DAPI_IDX, chan_names, chan_count), 0, chan_count-1))
    npm1_ch = int(np.clip(_resolve_channel(NPM1_NAME, NPM1_IDX, chan_names, chan_count), 0, chan_count-1))
    ncc_ch  = int(np.clip(_resolve_channel(nucleolar_component,  NC_IDX,  chan_names, chan_count), 0, chan_count-1))

    dapi = arr[dapi_ch].astype(np.float32)
    npm1 = arr[npm1_ch].astype(np.float32)
    nc  = arr[ncc_ch].astype(np.float32)

    nuclei_raw   = skio.imread(nuclei_labels_path)
    nucleoli_raw = skio.imread(npm1_labels_path)

    if nuclei_raw.ndim == 3:  # RGB -> binary -> labels
      nuclei = measure.label(rgb2gray(nuclei_raw) > 0).astype(np.int32)
    else:
      nuclei = measure.label(nuclei_raw > 0).astype(np.int32)

    if nucleoli_raw.ndim == 3:
      nucleoli = measure.label(rgb2gray(nucleoli_raw) > 0).astype(np.int32)
    else:
      nucleoli = measure.label(nucleoli_raw > 0).astype(np.int32)

# Confine nucleoli to nuclei and relabel
    nucleoli = np.where(nuclei > 0, nucleoli, 0)
    nucleoli = measure.label(nucleoli > 0).astype(np.int32)

    # ensure background is 0 and labels are contiguous-ish
    nuclei   = measure.label(nuclei > 0).astype(np.int32)
    nucleoli = np.where(nuclei > 0, nucleoli, 0)
    nucleoli = measure.label(nucleoli > 0).astype(np.int32)

    # ---------------- flatfield / basic images ----------------
    dapi_ff = flatfield_correct(dapi, sigma_px=30)
    npm1_ff = flatfield_correct(npm1, sigma_px=30)
    nc_ff  = flatfield_correct(nc,  sigma_px=30)

    #inner nucleus mask (remove rims)
    inner_nucleus_mask = morphology.erosion(nuclei > 0, morphology.disk(10))

    #FBL segmentation (foci inside inner nucleus)
    nc_in   = nc_ff.copy(); nc_in[nuclei == 0] = 0
    nc_bg   = filters.gaussian(nc_in, sigma=6.0)
    nc_corr = np.clip(nc_in - nc_bg, 0, None)

    nc_mask = np.zeros_like(nc_corr, dtype=bool)
    if nc_corr[nuclei > 0].size > 0:
        t_nc  = filters.threshold_otsu(nc_corr[nuclei > 0])
        nc_mask = (nc_corr > t_nc) & inner_nucleus_mask

    nc_labels = measure.label(nc_mask)
    parent_nc = parent_by_mode(nc_labels, nuclei)

    props_nc = measure.regionprops_table(
        nc_labels, intensity_image=nc_ff,
        properties=["label", "area", "mean_intensity", "eccentricity", "solidity", "perimeter"]
    )
    df_nc = pd.DataFrame(props_nc)
    if len(df_nc):
        df_nc["parent_nucleus"] = df_nc["label"].map(parent_nc).fillna(0).astype(int)
        df_nc = df_nc[df_nc["parent_nucleus"] > 0].copy()
    else:
        df_nc = pd.DataFrame(columns=["label","area","mean_intensity","eccentricity",
                                       "solidity","perimeter","parent_nucleus"])

    # ---------------- NPM1 mask for Manders (intensity-based) ----------------
    npm1_in   = npm1_ff.copy(); npm1_in[nuclei == 0] = 0
    npm1_bg   = filters.gaussian(npm1_in, sigma=6.0)
    npm1_corr = np.clip(npm1_in - npm1_bg, 0, None)

    npm1_mask = np.zeros_like(npm1_corr, dtype=bool)
    if npm1_corr[nuclei > 0].size > 0:
        t_npm1 = filters.threshold_otsu(npm1_corr[nuclei > 0])
        npm1_mask = (npm1_corr > t_npm1) & inner_nucleus_mask

    core_mask = inner_nucleus_mask

    # Global Manders / Jaccard
    # Jaccard on binary masks
    union = np.logical_or(npm1_mask, nc_mask)[core_mask].sum()
    inter = np.logical_and(npm1_mask, nc_mask)[core_mask].sum()
    jaccard_global = float(inter) / (float(union) + 1e-9) if union > 0 else np.nan

    # Manders on intensities
    A = npm1_ff
    B = nc_ff
    vA = A[core_mask].ravel().astype(np.float32)
    vB = B[core_mask].ravel().astype(np.float32)

    if vA.size > 10 and vB.size > 10:
        try:
            tA = filters.threshold_otsu(vA)
        except ValueError:
            tA = vA.mean()
        try:
            tB = filters.threshold_otsu(vB)
        except ValueError:
            tB = vB.mean()
    else:
        tA, tB = 0, 0

    maskA = (A > tA) & core_mask
    maskB = (B > tB) & core_mask

    sumA = float(A[core_mask].sum() + 1e-9)
    sumB = float(B[core_mask].sum() + 1e-9)

    M1_global = float(A[maskA & maskB].sum()) / sumA   # NPM1 in FBL+ pixels
    M2_global = float(B[maskA & maskB].sum()) / sumB   # FBL  in NPM1+ pixels

    #Per-nucleus Manders (NPM1 <-> FBL)
    per_nuc_rows = []
    nuc_labels = np.unique(nuclei); nuc_labels = nuc_labels[nuc_labels != 0]

    for lb in nuc_labels:
        mm = (nuclei == lb) & core_mask
        if mm.sum() < 20:
            continue

        a = A[mm].ravel().astype(np.float32)
        b = B[mm].ravel().astype(np.float32)
        if a.size < 10 or b.size < 10:
            continue

        try:
            tA_n = filters.threshold_otsu(a)
        except ValueError:
            tA_n = a.mean()
        try:
            tB_n = filters.threshold_otsu(b)
        except ValueError:
            tB_n = b.mean()

        mA = a.sum() + 1e-9
        mB = b.sum() + 1e-9

        M1_n = float(a[(a > tA_n) & (b > tB_n)].sum()) / mA
        M2_n = float(b[(b > tB_n) & (a > tA_n)].sum()) / mB

        per_nuc_rows.append({
            "nucleus_label": int(lb),
            f"manders_M1_NPM1_in_{nucleolar_component}": M1_n,
            f"manders_M2_{nucleolar_component}_in_NPM1": M2_n
        })

    df_manders = pd.DataFrame(per_nuc_rows)
    df_manders.to_csv(os.path.join(out_root, f"per_nucleus_manders_NPM1_{nucleolar_component}.csv"), index=False)

    parent_map = parent_by_mode(nucleoli, nuclei)

    props_nucleoli = measure.regionprops_table(
        nucleoli, intensity_image=npm1_ff,
        properties=["label", "area", "mean_intensity", "eccentricity", "solidity", "perimeter"]
    )
    df_nucleo = pd.DataFrame(props_nucleoli)
    if len(df_nucleo):
        df_nucleo["parent_nucleus"] = df_nucleo["label"].map(parent_map).fillna(0).astype(int)
        df_nucleo = df_nucleo[df_nucleo["parent_nucleus"] > 0].copy()

    counts = (df_nucleo.groupby("parent_nucleus")
              .size().rename("nucleoli_count").reset_index()
              .rename(columns={"parent_nucleus": "nucleus_label"})) if len(df_nucleo) else \
              pd.DataFrame(columns=["nucleus_label","nucleoli_count"])

    props_nuclei = measure.regionprops_table(
        nuclei, intensity_image=dapi_ff,
        properties=["label", "area", "mean_intensity", "eccentricity", "solidity"]
    )
    df_cells = (pd.DataFrame(props_nuclei)
                .rename(columns={"label": "nucleus_label"})
                .merge(counts, on="nucleus_label", how="left")
                .fillna({"nucleoli_count": 0}))
    df_cells["nucleoli_count"] = df_cells["nucleoli_count"].astype(int)

    if PIXEL_SIZE_UM:
        px_area = float(PIXEL_SIZE_UM) ** 2
        if len(df_cells):
            df_cells["area_um2"] = df_cells["area"] * px_area
        if len(df_nucleo):
            df_nucleo["area_um2"] = df_nucleo["area"] * px_area
        if len(df_nc):
            df_nc["area_um2"] = df_nc["area"] * px_area

    total_area_nucleoli = float(df_nucleo["area"].sum()) if len(df_nucleo) else 0.0
    total_area_nuclei   = float(df_cells["area"].sum())  if len(df_cells)  else 1e-9
    ratio = total_area_nucleoli / total_area_nuclei

    if save_png:
        _save_gray(dapi, os.path.join(out_root, "raw_DAPI.png"))
        _save_gray(npm1, os.path.join(out_root, "raw_NPM1.png"))
        _save_gray(nc,  os.path.join(out_root, f"raw_{nucleolar_component}.png"))

    if save_tiff:
        _save_label_tiff(nuclei,   os.path.join(out_root, "nuclei_labels.tif"))
        _save_label_tiff(nucleoli, os.path.join(out_root, "nucleoli_labels.tif"))
        _save_label_tiff(nc_labels, os.path.join(out_root, "nc_labels.tif"))
        skio.imsave(os.path.join(out_root, "npm1_mask.tif"),
                    (npm1_mask.astype(np.uint8)*255), check_contrast=False)
        skio.imsave(os.path.join(out_root, f"{nucleolar_component}_mask.tif"),
                    (nc_mask.astype(np.uint8)*255),  check_contrast=False)

    if save_numpy:
        np.save(os.path.join(out_root, "nuclei_labels.npy"), nuclei)
        np.save(os.path.join(out_root, "nucleoli_labels.npy"), nucleoli)
        np.save(os.path.join(out_root, f"{nucleolar_component}_labels.npy"), nc_labels.astype(np.int32))
        np.save(os.path.join(out_root, "npm1_mask.npy"), npm1_mask.astype(np.uint8))
        np.save(os.path.join(out_root, f"{nucleolar_component}_mask.npy"),  nc_mask.astype(np.uint8))

    if save_png:
        skio.imsave(os.path.join(out_root, "nuclei_labels.png"),
                    (color.label2rgb(nuclei,   bg_label=0, bg_color=(0,0,0))*255).astype(np.uint8),
                    check_contrast=False)
        skio.imsave(os.path.join(out_root, "nucleoli_labels.png"),
                    (color.label2rgb(nucleoli, bg_label=0, bg_color=(0,0,0))*255).astype(np.uint8),
                    check_contrast=False)
        skio.imsave(os.path.join(out_root, f"{nucleolar_component}_labels.png"),
                    (color.label2rgb(nc_labels, bg_label=0, bg_color=(0,0,0))*255).astype(np.uint8),
                    check_contrast=False)

        skio.imsave(os.path.join(out_root, "overlay_nuclei.png"),
                    (overlay_labels(dapi, nuclei)*255).astype(np.uint8), check_contrast=False)
        skio.imsave(os.path.join(out_root, "overlay_nucleoli.png"),
                    (overlay_labels(npm1, nucleoli)*255).astype(np.uint8), check_contrast=False)

        ov_npm1_mask = overlay_labels(npm1, (npm1_mask.astype(np.uint8))*1)
        ov_nc_mask  = overlay_labels(nc,  (nc_mask.astype(np.uint8))*1)
        skio.imsave(os.path.join(out_root, "overlay_NPM1_mask.png"),
                    (np.clip(ov_npm1_mask,0,1)*255).astype(np.uint8), check_contrast=False)
        skio.imsave(os.path.join(out_root, f"overlay_{nucleolar_component}_mask.png"),
                    (np.clip(ov_nc_mask,0,1)*255).astype(np.uint8), check_contrast=False)

    if save_figure:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0,0].imshow(normalize01(dapi), cmap="Blues");  axes[0,0].set_title("DAPI (raw)")
        axes[0,1].imshow(normalize01(npm1), cmap="Reds");   axes[0,1].set_title("NPM1 (raw)")
        axes[0,2].imshow(normalize01(nc),  cmap="Greens"); axes[0,2].set_title(f"{nucleolar_component} (raw)")
        axes[1,0].imshow(overlay_labels(dapi, nuclei));     axes[1,0].set_title(f"Nuclei ({nuclei.max()} labels)")
        axes[1,1].imshow(overlay_labels(npm1, nucleoli));   axes[1,1].set_title(f"Nucleoli ({nucleoli.max()} labels)")
        axes[1,2].imshow(overlay_labels(nc, nc_labels));  axes[1,2].set_title(f"{nucleolar_component} foci ({nc_labels.max()} labels)")
        for ax in axes.ravel():
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(out_root, "summary_2x3.png"), dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)

    # summary JSON
    meta = {
        "file": nd2_file, "group": group_name,
        "channels": {
            "DAPI": DAPI_NAME, "NPM1": NPM1_NAME, f"{nucleolar_component}": nucleolar_component,
            "DAPI_idx": dapi_ch, "NPM1_idx": npm1_ch, f"{nucleolar_component}_idx": ncc_ch
        },
        "n_nuclei": int(nuclei.max()),
        "n_nucleoli": int(nucleoli.max()),
        f"n_{nucleolar_component}_foci": int(nc_labels.max()),
        "total_area_nucleoli_px": total_area_nucleoli,
        "total_area_nuclei_px": total_area_nuclei,
        "nucleoli_over_nucleus_area_ratio": ratio,
        "pixel_size_um": PIXEL_SIZE_UM,
        f"coloc_global_NPM1_{nucleolar_component}": {
            "jaccard": jaccard_global,
            f"manders_M1_NPM1_in_{nucleolar_component}": M1_global,
            f"manders_M2_{nucleolar_component}_in_NPM1": M2_global
        }
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "out_dir": out_root,
        "nuclei_count": int(nuclei.max()),
        "nucleoli_count": int(nucleoli.max()),
        f"n_{nucleolar_component}_foci": int(nc_labels.max()),
        "ratio": ratio,
        f"coloc_global_NPM1_{nucleolar_component}": {
            "jaccard": jaccard_global,
            "M1": M1_global,
            "M2": M2_global
        },
        "cells_csv": os.path.join(out_root, "per_cell.csv"),
        "nucleoli_csv": os.path.join(out_root, "per_nucleolus.csv"),
        f"{nucleolar_component}_csv": os.path.join(out_root, f"per_{nucleolar_component}_focus.csv"),
        "per_nucleus_manders_csv": os.path.join(out_root, "per_nucleus_manders_NPM1_{nucleolar_component}.csv"),
        "summary_fig": os.path.join(out_root, "summary_2x3.png"),
    }
