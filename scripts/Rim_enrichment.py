import os
import numpy as np
import pandas as pd
from skimage import color, io as skio, measure, morphology
from utils import _ensure_dir


def analyze_rim_foci_enrichment(
    nd2_file,               
    nuclei_labels_path,     
    nucleoli_labels_path,   
    foci_mask_path,         
    target_intensity_path,  # <--- NEW ARGUMENT: Path to raw FBL/UBF image (tif or png)
    target_name="FBL",      
    rim_width_px=4,
    output_dir="./output_rim_foci",
    group_name="",
    save_visuals=True
):
    """
    Calculates Rim Enrichment based on PRE-SEGMENTED FOCI MASKS and RAW INTENSITY.
    Metrics:
      1. Area Enrichment: Do foci occupy more space in the rim?
      2. Intensity Enrichment: Is the protein concentration higher in the rim?
    """
    
    file_stem = os.path.splitext(os.path.basename(nd2_file))[0]
    prefix = f"{group_name}_" if group_name else ""
    out_root = _ensure_dir(os.path.join(output_dir, prefix + file_stem))

    def _load_lbl(p):
        l = skio.imread(p)
        l = np.squeeze(l)
        if l.ndim == 3 and (l.shape[-1] == 3 or l.shape[-1] == 4): 
            from skimage.color import rgb2gray
            l = (rgb2gray(l) > 0)
        if l.max() <= 1:
            l = measure.label(l > 0).astype(np.int32)
        return l.astype(np.int32)

    def _load_img(p):
        img = skio.imread(p)
        img = np.squeeze(img)
        # If RGB, take mean or first channel, but warn user
        if img.ndim == 3:
            return img[..., 0] # Assume channel 0 is relevant if passed as RGB
        return img

    nuclei = _load_lbl(nuclei_labels_path)
    nucleoli = _load_lbl(nucleoli_labels_path)
    
    # Load Foci Mask
    foci_mask_raw = skio.imread(foci_mask_path)
    foci_mask_raw = np.squeeze(foci_mask_raw)
    if foci_mask_raw.ndim == 3:
        foci_mask = (foci_mask_raw[..., 0] > 0).astype(bool)
    else:
        foci_mask = (foci_mask_raw > 0).astype(bool)

    #Load Intensity Image
    if target_intensity_path and os.path.exists(target_intensity_path):
        intensity_img = _load_img(target_intensity_path)
    else:
        print(f"Warning: Intensity path not found: {target_intensity_path}. Intensity metrics will be 0.")
        intensity_img = np.zeros_like(nuclei, dtype=float)

    #Map Nucleoli -> Nuclei
    parent_map = {}
    props_map = measure.regionprops(nucleoli, intensity_image=nuclei)
    for p in props_map:
        vals = p.image_intensity[p.image] 
        vals = vals[vals != 0]
        if vals.size > 0:
            parent_id = int(np.bincount(vals.astype(int)).argmax())
            parent_map[p.label] = parent_id

    #Iterate & Measure
    props = measure.regionprops(nucleoli)
    nucleolar_data = []

    if save_visuals:
        h, w = nuclei.shape
        vis_map = np.zeros((h, w, 3), dtype=np.float32)
        vis_map[foci_mask] = [0.4, 0.4, 0.4] 

    for prop in props:
        label_id = prop.label
        parent_id = parent_map.get(label_id, 0)
        if parent_id == 0: continue

        minr, minc, maxr, maxc = prop.bbox
        local_nuc_mask = nucleoli[minr:maxr, minc:maxc] == label_id
        local_foci     = foci_mask[minr:maxr, minc:maxc]
        local_int      = intensity_img[minr:maxr, minc:maxc] # <--- Crop intensity

        # Morphology
        selem = morphology.disk(rim_width_px)
        local_core_region = morphology.binary_erosion(local_nuc_mask, selem)
        local_rim_region  = np.logical_and(local_nuc_mask, ~local_core_region)

        # A. Area Metrics
        area_core_total = np.sum(local_core_region)
        area_rim_total  = np.sum(local_rim_region)
        foci_px_core    = np.sum(local_foci & local_core_region)
        foci_px_rim     = np.sum(local_foci & local_rim_region)

        # B. Intensity Metrics (Sum of pixel values in each region)
        # Note: We measure intensity of the WHOLE compartment (background + foci)
        # to see where the protein distributes generally.
        int_sum_core = np.sum(local_int[local_core_region])
        int_sum_rim  = np.sum(local_int[local_rim_region])

        nucleolar_data.append({
            "parent_nucleus": parent_id,
            "nucleolus_label": label_id,
            "area_core_total": area_core_total,
            "area_rim_total":  area_rim_total,
            "foci_px_core":    foci_px_core,
            "foci_px_rim":     foci_px_rim,
            "int_sum_core":    int_sum_core, # <--- NEW
            "int_sum_rim":     int_sum_rim   # <--- NEW
        })

        if save_visuals:
            vis_map[minr:maxr, minc:maxc, 0] += local_core_region * 0.4
            vis_map[minr:maxr, minc:maxc, 1] += local_rim_region * 0.4

    # --- 4. Aggregate Per Cell ---
    df_raw = pd.DataFrame(nucleolar_data)
    if df_raw.empty: return pd.DataFrame(), nuclei

    df_cell = df_raw.groupby("parent_nucleus").agg({
        "area_core_total": "sum",
        "area_rim_total":  "sum",
        "foci_px_core":    "sum",
        "foci_px_rim":     "sum",
        "int_sum_core":    "sum",
        "int_sum_rim":     "sum",
        "nucleolus_label": "count"
    }).rename(columns={"nucleolus_label": "nucleoli_count"}).reset_index()

    # --- 5. Calculate Metrics ---

    # --- METRIC SET 1: AREA (Spatial) ---
    df_cell["total_foci_area"] = df_cell["foci_px_core"] + df_cell["foci_px_rim"]
    
    # Proportion of Foci Area in Rim
    df_cell["prop_foci_area_in_rim"] = df_cell.apply(
        lambda r: r["foci_px_rim"] / r["total_foci_area"] if r["total_foci_area"] > 0 else 0, axis=1)

    # Area Density (Crowding)
    df_cell["density_area_core"] = df_cell["foci_px_core"] / df_cell["area_core_total"].replace(0, np.nan)
    df_cell["density_area_rim"]  = df_cell["foci_px_rim"]  / df_cell["area_rim_total"].replace(0, np.nan)
    df_cell[f"score_area_{target_name}"] = df_cell["density_area_rim"] / df_cell["density_area_core"]

    # --- METRIC SET 2: INTENSITY (Concentration) ---
    df_cell["total_intensity"] = df_cell["int_sum_core"] + df_cell["int_sum_rim"]

    # Proportion of Signal in Rim ("Mass Fraction")
    df_cell["prop_intensity_in_rim"] = df_cell.apply(
        lambda r: r["int_sum_rim"] / r["total_intensity"] if r["total_intensity"] > 0 else 0, axis=1)

    # Mean Intensity (Concentration)
    df_cell["mean_int_core"] = df_cell["int_sum_core"] / df_cell["area_core_total"].replace(0, np.nan)
    df_cell["mean_int_rim"]  = df_cell["int_sum_rim"]  / df_cell["area_rim_total"].replace(0, np.nan)

    # Intensity Ratio (Enrichment Score)
    # > 1.0 means the protein is brighter/more concentrated in the rim
    df_cell[f"score_intensity_{target_name}"] = df_cell["mean_int_rim"] / df_cell["mean_int_core"]

    # --- 6. Save ---
    df_cell.to_csv(os.path.join(out_root, f"rim_enrichment_{target_name}.csv"), index=False)

    if save_visuals:
        vis_map = np.clip(vis_map, 0, 1)
        skio.imsave(
            os.path.join(out_root, f"vis_{target_name}_rim.png"),
            (vis_map * 255).astype(np.uint8),
            check_contrast=False
        )
        
    return df_cell, nuclei