import os, fnmatch, glob, zipfile, shutil, warnings
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================
# Settings / paths
# =========================
dir_flux = "/burg/glab/users/rg3390/data/FLUXNET2015/"
fn_fluxlist = "fluxsite_info_metalist.xlsx"
out_dir = "/burg/glab/users/rg3390/data/FLUXNET2015/"
dir_SOS_EOS = "/burg/glab/users/rg3390/data/FLUXNET2015/SOS_EOS/"

# Output filenames
fn_neg_H = "df_neg_H_v10_rainbet12hrs_noZL.csv"
fn_neg_H_consec2hr = "df_neg_H_consec2hr_v10_measured_only_rainbet12hrs_nozL.csv"
fn_site_summary = "site_valid_negH_summary_v10_measured_only_rainbet12hrs_nozL.csv"

# =========================
# Physical constants
# =========================
es0 = 6.11           # hPa
T0 = 273.15          # K
Lv = 2.5e6           # J/kg
Rv = 461.0           # J/K/kg
Rd = 287.0           # J/K/kg
epson = Rd / Rv
Cp = 1004.0          # J/K/kg
vK_const = 0.4       # von Karman constant
g = 9.8              # m/s^2

# =========================
# Columns & QC maps
# =========================
# FLUXNET / AmeriFlux / ICOS
pick_cols = [ \
    "TIMESTAMP_START","TIMESTAMP_END","P_F","P_F_QC","LE_CORR","LE_F_MDS","LE_F_MDS_QC", \
    "H_CORR","H_F_MDS","H_F_MDS_QC","TA_F","TA_F_QC","PA_F","PA_F_QC", \
    "GPP_NT_VUT_REF","GPP_DT_VUT_REF","NEE_VUT_REF_QC","VPD_F","VPD_F_QC", \
    "SW_IN_F","SW_IN_F_QC","WS_F","WS_F_QC","USTAR","LW_IN_F","LW_IN_F_QC", \
    "NETRAD","G_F_MDS","G_F_MDS_QC"
]
dic_QC1 = { \
    "P_F":"P_F_QC","LE_CORR":"LE_F_MDS_QC","LE_F_MDS":"LE_F_MDS_QC", \
    "H_CORR":"H_F_MDS_QC","H_F_MDS":"H_F_MDS_QC","TA_F":"TA_F_QC", \
    "PA_F":"PA_F_QC","GPP_NT_VUT_REF":"NEE_VUT_REF_QC","GPP_DT_VUT_REF":"NEE_VUT_REF_QC", \
    "VPD_F":"VPD_F_QC","SW_IN_F":"SW_IN_F_QC","WS_F":"WS_F_QC","LW_IN_F":"LW_IN_F_QC", \
    "G_F_MDS":"G_F_MDS_QC"
}
dic_QC2 = { \
    "P_F":"P_F_QC","LE_flux":"LE_F_MDS_QC","H_flux":"H_F_MDS_QC","TA_F":"TA_F_QC", \
    "PA_F":"PA_F_QC","GPP_NT_VUT_REF":"NEE_VUT_REF_QC","GPP_DT_VUT_REF":"NEE_VUT_REF_QC", \
    "VPD_F":"VPD_F_QC","SW_IN_F":"SW_IN_F_QC","WS_F":"WS_F_QC","LW_IN_F":"LW_IN_F_QC", \
    "G_flux":"G_F_MDS_QC" \
}
# only these are replaced by QC3-cleaned values
dic_QC_flux = ["GPP_DT_VUT_REF","LE_flux","H_flux","G_flux","NETRAD","GPP_NT_VUT_REF"]
var_to_QC = [ \
    "P_F","LE_CORR","LE_F_MDS","H_CORR","H_F_MDS","TA_F","PA_F","GPP_NT_VUT_REF", \
    "GPP_DT_VUT_REF","VPD_F","SW_IN_F","WS_F","LW_IN_F","G_F_MDS" \
]
drop_cols = [ \
    "P_F_QC","LE_CORR","LE_F_MDS","LE_F_MDS_QC","H_CORR","H_F_MDS","H_F_MDS_QC", \
    "TA_F_QC","PA_F_QC","NEE_VUT_REF_QC","VPD_F_QC","SW_IN_F_QC","WS_F_QC","LW_IN_F_QC", \
    "G_F_MDS","G_F_MDS_QC","GPP_NT_VUT_REF","GPP_DT_VUT_REF","time_year" \
]

# OzFlux
pick_cols_Ozflux = [ \
    "TIMESTAMP_START","TIMESTAMP_END","P_F","P_F_QC","LE_flux","LE_flux_QC","H_flux","H_flux_QC", \
    "TA_F","TA_F_QC","PA_F","PA_F_QC","GPP_NT_VUT_REF","GPP_DT_VUT_REF", \
    "GPP_NT_VUT_REF_QC","GPP_DT_VUT_REF_QC","VPD_F","VPD_F_QC","SW_IN_F","SW_IN_F_QC", \
    "WS_F","WS_F_QC","LW_IN_F","LW_IN_F_QC","NETRAD","NETRAD_QC","G_flux","G_flux_QC", \
    "USTAR","USTAR_QC","GPP","GPP_QC" \
]
dic_QC_Ozflux = { \
    "P_F":"P_F_QC","LE_flux":"LE_flux_QC","H_flux":"H_flux_QC","TA_F":"TA_F_QC", \
    "PA_F":"PA_F_QC","VPD_F":"VPD_F_QC","SW_IN_F":"SW_IN_F_QC","WS_F":"WS_F_QC", \
    "LW_IN_F":"LW_IN_F_QC","NETRAD":"NETRAD_QC","G_flux":"G_flux_QC","USTAR":"USTAR_QC","GPP":"GPP_QC" \
}
var_QC_Ozflux_flux = ["LE_flux","H_flux","G_flux","NETRAD","GPP"]
var_to_QC_Ozflux = ["P_F","LE_flux","H_flux","TA_F","PA_F","VPD_F","SW_IN_F","WS_F","LW_IN_F","NETRAD","G_flux","USTAR","GPP"]
drop_cols_Ozflux = [ \
    "P_F_QC","LE_flux_QC","H_flux_QC","TA_F_QC","PA_F_QC","GPP_NT_VUT_REF_QC", \
    "GPP_DT_VUT_REF_QC","VPD_F_QC","SW_IN_F_QC","WS_F_QC","LW_IN_F_QC","NETRAD_QC", \
    "G_flux_QC","USTAR_QC","GPP_QC","GPP_NT_VUT_REF","GPP_DT_VUT_REF" \
]

# LBA-ECO
pick_cols_LBAECO = [ \
    "TIMESTAMP_END","P_F","P_F_QC","LE_flux","LE_flux_QC","H_flux","H_flux_QC", \
    "TA_F","TA_F_QC","PA_F","PA_F_QC","GPP_NT_VUT_REF","GPP_NT_VUT_REF_QC","VPD_F","VPD_F_QC", \
    "SW_IN_F","SW_IN_F_QC","WS_F","WS_F_QC","LW_IN_F","LW_IN_F_QC","NETRAD","NETRAD_QC", \
    "G_flux","G_flux_QC","USTAR","USTAR_QC","GPP","GPP_QC" \
]
var_to_QC3_LBAECO = ["GPP","LE_flux","H_flux","NETRAD","G_flux"]
drop_cols_LBAECO = [ \
    "P_F_QC","LE_flux_QC","H_flux_QC","TA_F_QC","PA_F_QC","GPP_NT_VUT_REF_QC","VPD_F_QC", \
    "SW_IN_F_QC","WS_F_QC","LW_IN_F_QC","NETRAD_QC","G_flux_QC","USTAR_QC","GPP_QC" \
]

# =========================
# Helper: outlier removal (unchanged logic, vectorized where possible)
# =========================
def remove_outlier(df, var):
    """
    Sliding ±7-day window, same-time-of-day (±2h) distribution.
    Flag outliers as |x - mean| > k*std, where k=5 for cold (TA_5day_mean<5C), else 3.
    Only enforce for flux-like vars (in dic_QC_flux). If not flux, drop when cold.
    """
    df_no_outlier = df.reset_index(drop=True).copy()
    outlier_mask = np.zeros(len(df_no_outlier), dtype=bool)

    # hour at TIMESTAMP_START; if missing, derive from END - this code assumes both present
    if "hour_start" not in df_no_outlier.columns:
        df_no_outlier["hour_start"] = df_no_outlier["TIMESTAMP_START"].dt.hour

    # precompute for speed
    ts_end = df_no_outlier["TIMESTAMP_END"].values
    hours  = df_no_outlier["hour_start"].values

    for idx in range(len(df_no_outlier)):
        center_time = ts_end[idx]
        h0 = hours[idx]
        window_start = center_time - np.timedelta64(7, "D")
        window_end   = center_time + np.timedelta64(7, "D")

        # mask rows within ±7d and within ±2h of local hour (mod 24)
        hour_diff = (hours - h0) % 24
        in_win = (ts_end >= window_start) & (ts_end <= window_end)
        same_time = (hour_diff <= 2)
        valid = in_win & same_time & (~pd.isna(df_no_outlier[var].values))

        window_values = df_no_outlier.loc[valid, var]
        if len(window_values) < 5:
            continue

        mean_val = window_values.mean()
        std_val  = window_values.std()

        # 5-day mean T to switch threshold
        tmask = (ts_end >= center_time - np.timedelta64(2, "D")) & (ts_end <= center_time + np.timedelta64(2, "D"))
        if "TA_low" in df_no_outlier.columns:
            TA_5day_mean = df_no_outlier.loc[tmask, "TA_low"].mean()
        else:
            TA_5day_mean = 10.0

        k = 5.0 if (TA_5day_mean < 5.0) else 3.0
        upper = mean_val + k * std_val
        lower = mean_val - k * std_val

        x = df_no_outlier.at[idx, var]
        if var in dic_QC_flux:
            if (x > upper) or (x < lower):
                outlier_mask[idx] = True
        else:
            # non-flux in cold: drop
            if TA_5day_mean < 5.0:
                outlier_mask[idx] = True

    df_no_outlier.loc[outlier_mask, var] = np.nan
    return df_no_outlier


# =========================
# Worker
# =========================
def process_and_plot(args):
    site_ID, site_name, site_year_start, site_year_end, site_IGBP, site_src, z_measure, site_lat, site_lon, site_elv = args
    print(f"in process_and_plot {site_ID} and source from {site_src}")

    # -------------------------
    # Choose directories/patterns
    # -------------------------
    if site_src == "ICOS":
        zip_dir = "/burg/glab/users/rg3390/data/FLUXNET2015/ICOS/warm_winter_2020_release_2022/"
        pwd_dir = os.listdir(zip_dir)
        pattern = f"FLX_{site_ID}_*_FULLSET_*.zip"
    elif site_src == "AmeriFlux":
        zip_dir = "/burg/glab/users/rg3390/data/FLUXNET2015/AmeriFlux/"
        pwd_dir = os.listdir(zip_dir)
        pattern = f"AMF_{site_ID}_*_FULLSET_*.zip"
    elif site_src == "FLUXNET":
        zip_dir = "/burg/glab/users/rg3390/data/FLUXNET2015/FLUXNET2015_zipped/"
        pwd_dir = os.listdir(zip_dir)
        pattern = f"FLX_{site_ID}_*_FULLSET_*.zip"
    elif site_src == "Ozflux":
        nc_dir = "/burg/glab/users/rg3390/data/FLUXNET2015/OzFlux/"
        pattern = site_name + "_L6.nc"
    elif site_src == "LBA-ECO":
        xlsx_dir = "/burg/glab/users/rg3390/data/FLUXNET2015/LBA-ECO/"
        pattern = site_ID + "_CfluxBF.xlsx"
    else:
        print(f"[{site_ID}] Unknown source: {site_src}")
        return (pd.DataFrame(), pd.DataFrame(), 0, 0, 0)

    # -------------------------
    # Branch by source
    # -------------------------
    if site_src in ("ICOS", "AmeriFlux", "FLUXNET"):
        # ---- Find zip file once ----
        zip_path = None
        for entry in pwd_dir:
            if fnmatch.fnmatch(entry, pattern):
                zip_path = os.path.join(zip_dir, entry)
                break
        if zip_path is None:
            print(f"[{site_ID}] No matching ZIP found.")
            return (pd.DataFrame(), pd.DataFrame(), 0, 0, 0)

        # ---- Read target CSV directly from zip (no extract) ----
        with zipfile.ZipFile(zip_path, "r") as zf:
            hh_names = [n for n in zf.namelist() if ("FULLSET_HH" in n and site_ID in n and n.endswith(".csv"))]
            hr_names = [n for n in zf.namelist() if ("FULLSET_HR" in n and site_ID in n and n.endswith(".csv"))]
            if not (hh_names or hr_names):
                print(f"[{site_ID}] no FULLSET_HH/HR CSV inside {os.path.basename(zip_path)}")
                return (pd.DataFrame(), pd.DataFrame(), 0, 0, 0)

            member_name = hh_names[0] if hh_names else hr_names[0]
            len_HH = 1 if "FULLSET_HH" in member_name else 0
            day_timestep = 48 if len_HH else 24
            rain_QC_timestep = 24 if len_HH else 12

            with zf.open(member_name) as fh:
                flux_df = pd.read_csv(fh, usecols=lambda x: x in pick_cols)

        # ---- Time columns ----
        flux_df["TIMESTAMP_START"] = pd.to_datetime(flux_df["TIMESTAMP_START"], format="%Y%m%d%H%M")
        flux_df["TIMESTAMP_END"]   = pd.to_datetime(flux_df["TIMESTAMP_END"],   format="%Y%m%d%H%M")
        flux_df["time_year"]       = flux_df["TIMESTAMP_START"].dt.year
        flux_df["time_DOY"]        = flux_df["TIMESTAMP_START"].dt.dayofyear

        # ---- Basic QC & conversions ----
        flux_df.replace(-9999, np.nan, inplace=True)
        flux_df["PA_F"] = flux_df["PA_F"] * 10.0  # kPa -> hPa

        # keep good (0,1) if QC flag exists; else keep as-is
        for var in var_to_QC:
            qc_col = dic_QC1[var]
            if qc_col in flux_df.columns:
                keep = flux_df[qc_col].isin([0]) | flux_df[qc_col].isna()
                flux_df.loc[~keep, var] = np.nan

        # choose H/LE flux columns
        flux_df["H_flux"]  = np.where(flux_df["H_CORR"].isna(),  flux_df["H_F_MDS"],  flux_df["H_CORR"])
        flux_df["LE_flux"] = np.where(flux_df["LE_CORR"].isna(), flux_df["LE_F_MDS"], flux_df["LE_CORR"])
        flux_df["G_flux"]  = flux_df["G_F_MDS"]

        # RH
        e_sat = es0 * np.exp((Lv / Rv) * ((1 / T0) - (1 / (flux_df["TA_F"] + 273.15))))
        flux_df["e_sat"] = e_sat
        flux_df["RH"] = 100.0 * (flux_df["e_sat"] - flux_df["VPD_F"]) / flux_df["e_sat"]
        flux_df.loc[~flux_df["RH"].between(0, 100), "RH"] = np.nan

        # Vectorized rainy-event screen (previous 12h or 24 half-hours)
        flux_df["P_F"] = pd.to_numeric(flux_df["P_F"], errors="coerce")
        win = (48 if len_HH else 24) + 1
        rain_sum = flux_df["P_F"].rolling(win, center=True,  min_periods=1).sum()
        rain_hit = rain_sum > 0
        cols_nan = ["USTAR","H_flux","LE_flux","GPP_NT_VUT_REF","GPP_DT_VUT_REF","G_flux"]
        cols_nan = [c for c in cols_nan if c in flux_df.columns]
        if cols_nan:
            flux_df.loc[rain_hit, cols_nan] = np.nan

        # Air density
        mixing_ratio = (epson * (flux_df["e_sat"] - flux_df["VPD_F"])) / (flux_df["PA_F"] - (flux_df["e_sat"] - flux_df["VPD_F"]))
        q = mixing_ratio / (1 + mixing_ratio)
        flux_df["rho_air"] = flux_df["PA_F"] * 100.0 / (Rd * (1 + 0.608 * q) * (flux_df["TA_F"] + 273.15))

        # USTAR screening (lower/upper)
        USTAR_mean = flux_df["USTAR"].mean(skipna=True)
        USTAR_std  = flux_df["USTAR"].std(skipna=True)
        USTAR_upper = USTAR_mean + 2 * USTAR_std
        flux_df.loc[~((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)), "USTAR"] = np.nan

        # ---- QC3 outlier removal on selected vars ----
        site_data = pd.DataFrame({"TIMESTAMP_START": flux_df["TIMESTAMP_START"], "TIMESTAMP_END": flux_df["TIMESTAMP_END"]})
        flux_df["SW_IN_F"] = flux_df["SW_IN_F"]
        for var, qc in dic_QC2.items():
            if var not in flux_df.columns:  # skip missing
                continue
            df_var = flux_df[["TIMESTAMP_START","TIMESTAMP_END","SW_IN_F",var]].copy()
            # for freezing logic
            df_var["TA_low"]    = flux_df["TA_F"]
            df_var["TA_low_QC"] = flux_df["TA_F_QC"]
            if df_var["TA_low"].notna().any():
                df_var["TA_low"] = df_var["TA_low"].where(df_var["TA_low_QC"].isin([0]), np.nan)
            df_var.drop(columns=["TA_low_QC"], inplace=True)

            if var in dic_QC_flux:
                df_no = remove_outlier(df_var, var)
                df_no.drop(columns=["TA_low","hour_start"], inplace=True, errors="ignore")
            else:
                df_no = df_var.drop(columns=["TA_low"])
            site_data = site_data.merge(df_no[["TIMESTAMP_START","TIMESTAMP_END",var]], on=["TIMESTAMP_START","TIMESTAMP_END"], how="left")

        # Replace ONLY selected target vars with QC3-cleaned values
        KEYS = ["TIMESTAMP_START","TIMESTAMP_END"]
        targets = [c for c in dic_QC_flux if c in site_data.columns]
        if targets:
            site_data = site_data.drop_duplicates(subset=KEYS, keep="last")
            f_idx = flux_df.set_index(KEYS)
            s_idx = site_data.set_index(KEYS)
            # bring missing
            miss = [c for c in targets if c not in f_idx.columns]
            if miss:
                f_idx = f_idx.join(s_idx[miss], how="left")
            exist = [c for c in targets if c in f_idx.columns]
            if exist:
                f_idx.update(s_idx[exist], overwrite=True)
            flux_df = f_idx.reset_index()

        # ---- Diagnostics ----
        flux_df["theta"] = (flux_df["TA_F"] + 273.15) * ((1000.0 / flux_df["PA_F"]) ** (Rd / Cp))
        if np.isnan(z_measure):
            flux_df["z_L"] = np.nan
        else:
            flux_df["z_L"] = -(z_measure * vK_const * g * flux_df["H_flux"]) / ((flux_df["USTAR"] ** 3) * flux_df["theta"] * flux_df["rho_air"] * Cp)

        flux_df["energy_closure"] = (flux_df["LE_flux"] + flux_df["H_flux"]) / (flux_df["NETRAD"] - flux_df["G_flux"])
        if "G_flux" in flux_df.columns:
            flux_df["energy_closure"] = np.where(
                flux_df["energy_closure"].isna() & flux_df["G_flux"].isna(),
                (flux_df["LE_flux"] + flux_df["H_flux"]) / flux_df["NETRAD"],
                flux_df["energy_closure"]
            )
        else:
            flux_df["energy_closure"] = (flux_df["LE_flux"] + flux_df["H_flux"]) / flux_df["NETRAD"]

        # Use DT as GPP
        flux_df["GPP"] = flux_df["GPP_DT_VUT_REF"]

        # ---- SOS/EOS filter ----
        sos_path = os.path.join(dir_SOS_EOS, f"{site_ID}_SOS_EOS.csv")
        sos_eos_df = pd.read_csv(sos_path).replace(-9999, np.nan)
        if not np.issubdtype(sos_eos_df["year"].dtype, np.integer):
            sos_eos_df["year"] = sos_eos_df["year"].astype("Int64")

        season_mask = pd.Series(False, index=flux_df.index)
        for _, row in sos_eos_df.dropna(subset=["year"]).iterrows():
            y = int(row["year"])
            w1 = w2 = None
            if pd.notna(row["SOS_1_GPP"]) and pd.notna(row["EOS_1_GPP"]):
                sos1, eos1 = int(row["SOS_1_GPP"]), int(row["EOS_1_GPP"])
                if sos1 <= eos1:
                    w1 = (flux_df["time_year"].eq(y) & flux_df["time_DOY"].between(sos1, eos1))
                else:
                    prev_part = (flux_df["time_year"].eq(y - 1) & (flux_df["time_DOY"] >= sos1))
                    curr_part = (flux_df["time_year"].eq(y) & (flux_df["time_DOY"] <= eos1))
                    w1 = prev_part | curr_part
            if pd.notna(row["SOS_2_GPP"]) and pd.notna(row["EOS_2_GPP"]):
                sos2, eos2 = int(row["SOS_2_GPP"]), int(row["EOS_2_GPP"])
                if sos2 <= eos2:
                    w2 = (flux_df["time_year"].eq(y) & flux_df["time_DOY"].between(sos2, eos2))
                else:
                    w2 = (flux_df["time_year"].eq(y) & ((flux_df["time_DOY"] >= sos2) | (flux_df["time_DOY"] <= eos2)))
            if (w1 is not None) and (w2 is not None):
                season_mask |= (w1 | w2)
            elif w1 is not None:
                season_mask |= w1
            elif w2 is not None:
                season_mask |= w2

        flux_df = flux_df[season_mask]

        # ---- Site metadata, cleanup ----
        flux_df["site_ID"] = site_ID
        flux_df["site_name"] = site_name
        flux_df["site_lat"] = site_lat
        flux_df["site_lon"] = site_lon
        flux_df["site_elv"] = site_elv
        flux_df["IGBP"] = site_IGBP
        flux_df["site_src"] = site_src
        flux_df.replace("", np.nan, inplace=True)
        flux_df.drop(columns=[c for c in drop_cols if c in flux_df.columns], inplace=True, errors="ignore")

        # ---- Atmospheric selection ----
        if site_elv <= 500:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) & 
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) & 
                    (flux_df["LE_flux"] >= 0) & 
                    (flux_df["TA_F"] >= 5) & 
                    ((flux_df["PA_F"] < 1020) & (flux_df["PA_F"] > 940))
                    )
        elif 500 < site_elv <= 1000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 870)
                    )
        elif 1000 < site_elv <= 2000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 770)
                    )
        elif 2000 < site_elv <= 3000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 670)
                    )
        else:  # site_elv > 3000
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5)
                    )

        flux_df_atm_sel = flux_df[day_atm_mask]

        # valid count
        count_valid = int(len(flux_df_atm_sel))

        # negative H selection
        neg = flux_df_atm_sel["H_flux"] < 0
        count_neg_H_tmp = int(neg.sum())

        # consecutive 2-hr windows
        delta = pd.Timedelta(minutes=30) if len_HH == 1 else pd.Timedelta(hours=1)
        times = flux_df_atm_sel["TIMESTAMP_END"]
        step_ok = times.diff().eq(delta) & neg
        # group increases whenever either step is broken or current isn't neg
        grp = (~step_ok).cumsum()
        run_sizes = grp.map(grp.value_counts())
        need = 4 if len_HH == 1 else 2
        neg_H_df_sel_consec2hr = flux_df_atm_sel[neg & (run_sizes >= need)]
        count_neg_H_consec2hr_tmp = int((run_sizes[neg] >= need).sum())

        # extract simple neg_H table (all negative points meeting atm mask)
        neg_H_df_sel = flux_df_atm_sel[neg]

        print(
            "finish searching site = ", site_ID,
            ", valid =", count_valid,
            ", number of negative H =", count_neg_H_tmp,
            ", number of 2hr negative H =", count_neg_H_consec2hr_tmp
        )

        return (neg_H_df_sel, neg_H_df_sel_consec2hr, count_valid, count_neg_H_tmp, count_neg_H_consec2hr_tmp)

    elif site_src == "Ozflux":
        # --------------- OzFlux (NetCDF) ---------------
        flux_ds = xr.open_dataset(os.path.join(nc_dir, pattern))
        flux_df = pd.DataFrame(index=np.arange(flux_ds.dims["time"]))
        flux_df["TIMESTAMP_END"] = pd.to_datetime(np.squeeze(flux_ds["time"].values))
        flux_df["P_F"]       = np.squeeze(flux_ds["Precip"].values)
        flux_df["P_F_QC"]    = np.squeeze(flux_ds["Precip_QCFlag"].values)
        flux_df["LE_flux"]   = np.squeeze(flux_ds["Fe"].values)
        flux_df["LE_flux_QC"]= np.squeeze(flux_ds["Fe_QCFlag"].values)
        flux_df["H_flux"]    = np.squeeze(flux_ds["Fh"].values)
        flux_df["H_flux_QC"] = np.squeeze(flux_ds["Fh_QCFlag"].values)
        flux_df["TA_F"]      = np.squeeze(flux_ds["Ta"].values)
        flux_df["TA_F_QC"]   = np.squeeze(flux_ds["Ta_QCFlag"].values)
        flux_df["PA_F"]      = np.squeeze(flux_ds["ps"].values)
        flux_df["PA_F_QC"]   = np.squeeze(flux_ds["ps_QCFlag"].values)
        if "GPP_LL" in flux_ds:
            flux_df["GPP"]    = np.squeeze(flux_ds["GPP_LL"].values)
            flux_df["GPP_QC"] = np.squeeze(flux_ds["GPP_LL_QCFlag"].values)
        else:
            flux_df["GPP"]    = np.squeeze(flux_ds["GPP_SOLO"].values)
            flux_df["GPP_QC"] = np.squeeze(flux_ds["GPP_SOLO_QCFlag"].values)
        flux_df["VPD_F"]     = np.squeeze(flux_ds["VPD"].values)
        flux_df["VPD_F_QC"]  = np.squeeze(flux_ds["VPD_QCFlag"].values)
        flux_df["SW_IN_F"]   = np.squeeze(flux_ds["Fsd"].values)
        flux_df["SW_IN_F_QC"]= np.squeeze(flux_ds["Fsd_QCFlag"].values)
        flux_df["WS_F"]      = np.squeeze(flux_ds["Ws"].values)
        flux_df["WS_F_QC"]   = np.squeeze(flux_ds["Ws_QCFlag"].values)
        flux_df["LW_IN_F"]   = np.squeeze(flux_ds["Fld"].values)
        flux_df["LW_IN_F_QC"]= np.squeeze(flux_ds["Fld_QCFlag"].values)
        flux_df["NETRAD"]    = np.squeeze(flux_ds["Fn"].values)
        flux_df["NETRAD_QC"] = np.squeeze(flux_ds["Fn_QCFlag"].values)
        flux_df["G_flux"]    = np.squeeze(flux_ds["Fg"].values)
        flux_df["G_flux_QC"] = np.squeeze(flux_ds["Fg_QCFlag"].values)
        flux_df["USTAR"]     = np.squeeze(flux_ds["ustar"].values)
        flux_df["USTAR_QC"]  = np.squeeze(flux_ds["ustar_QCFlag"].values)
        flux_ds.close(); del flux_ds

        # units, timing
        flux_df["PA_F"] = flux_df["PA_F"] * 10.0
        flux_df["VPD_F"] = flux_df["VPD_F"] * 10.0
        if (flux_df["TIMESTAMP_END"].iloc[1] - flux_df["TIMESTAMP_END"].iloc[0]) == pd.Timedelta(minutes=30):
            len_HH = 1
            flux_df["TIMESTAMP_START"] = flux_df["TIMESTAMP_END"] - pd.Timedelta(minutes=30)
        else:
            len_HH = 0
            flux_df["TIMESTAMP_START"] = flux_df["TIMESTAMP_END"] - pd.Timedelta(minutes=60)
        day_timestep = 48 if len_HH else 24
        rain_QC_timestep = 24 if len_HH else 12

        # QC1
        flux_df.replace(-9999, np.nan, inplace=True)
        for var in var_to_QC_Ozflux:
            qc = dic_QC_Ozflux[var]
            if qc in flux_df:
                keep = flux_df[qc].isin([0]) | flux_df[qc].isna()
                flux_df.loc[~keep, var] = np.nan

        # RH
        e_sat = es0 * np.exp((Lv / Rv) * ((1 / T0) - (1 / (flux_df["TA_F"] + 273.15))))
        flux_df["e_sat"] = e_sat
        flux_df["RH"] = 100.0 * (flux_df["e_sat"] - flux_df["VPD_F"]) / flux_df["e_sat"]
        flux_df.loc[~flux_df["RH"].between(0, 100), "RH"] = np.nan

        # Vectorized rainy-event screen
        flux_df["P_F"] = pd.to_numeric(flux_df["P_F"], errors="coerce")
        win = (48 if len_HH else 24) + 1
        rain_sum = flux_df["P_F"].rolling(win, min_periods=1).sum()
        rain_hit = rain_sum > 0
        cols_nan = ["USTAR","H_flux","LE_flux","GPP","G_flux"]
        cols_nan = [c for c in cols_nan if c in flux_df.columns]
        if cols_nan:
            flux_df.loc[rain_hit, cols_nan] = np.nan

        # Air density
        mixing_ratio = (epson * (flux_df["e_sat"] - flux_df["VPD_F"])) / (flux_df["PA_F"] - (flux_df["e_sat"] - flux_df["VPD_F"]))
        q = mixing_ratio / (1 + mixing_ratio)
        flux_df["rho_air"] = flux_df["PA_F"] * 100.0 / (Rd * (1 + 0.608 * q) * (flux_df["TA_F"] + 273.15))

        # USTAR bounds
        USTAR_mean = flux_df["USTAR"].mean(skipna=True); USTAR_std = flux_df["USTAR"].std(skipna=True)
        USTAR_upper = USTAR_mean + 2 * USTAR_std
        flux_df.loc[~((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)), "USTAR"] = np.nan

        # QC3 outlier removal
        site_data = pd.DataFrame({"TIMESTAMP_START": flux_df["TIMESTAMP_START"], "TIMESTAMP_END": flux_df["TIMESTAMP_END"]})
        flux_df["time_year"] = flux_df["TIMESTAMP_START"].dt.year
        flux_df["time_DOY"]  = flux_df["TIMESTAMP_START"].dt.dayofyear
        flux_df["SW_IN_F"]   = flux_df["SW_IN_F"]
        for var, qc in dic_QC_Ozflux.items():
            if var not in flux_df.columns:
                continue
            df_var = flux_df[["TIMESTAMP_START","TIMESTAMP_END","SW_IN_F",var]].copy()
            df_var["TA_low"] = flux_df["TA_F"]
            df_var["TA_low_QC"] = flux_df["TA_F_QC"]
            if df_var["TA_low"].notna().any():
                df_var["TA_low"] = df_var["TA_low"].where(df_var["TA_low_QC"].isin([0]), np.nan)
            df_var.drop(columns=["TA_low_QC"], inplace=True)
            if var in var_QC_Ozflux_flux:
                df_no = remove_outlier(df_var, var)
                df_no.drop(columns=["TA_low","hour_start"], inplace=True, errors="ignore")
            else:
                df_no = df_var.drop(columns=["TA_low"])
            site_data = site_data.merge(df_no[["TIMESTAMP_START","TIMESTAMP_END",var]], on=["TIMESTAMP_START","TIMESTAMP_END"], how="left")

        # Replace QC3-cleaned targets
        KEYS = ["TIMESTAMP_START","TIMESTAMP_END"]
        targets = [c for c in var_QC_Ozflux_flux if c in site_data.columns]
        if targets:
            site_data = site_data.drop_duplicates(subset=KEYS, keep="last")
            f_idx = flux_df.set_index(KEYS); s_idx = site_data.set_index(KEYS)
            miss = [c for c in targets if c not in f_idx.columns]
            if miss: f_idx = f_idx.join(s_idx[miss], how="left")
            exist = [c for c in targets if c in f_idx.columns]
            if exist: f_idx.update(s_idx[exist], overwrite=True)
            flux_df = f_idx.reset_index()

        # Diagnostics
        flux_df["theta"] = (flux_df["TA_F"] + 273.15) * ((1000.0 / flux_df["PA_F"]) ** (Rd / Cp))
        if np.isnan(z_measure):
            flux_df["z_L"] = np.nan
        else:
            flux_df["z_L"] = -(z_measure * vK_const * g * flux_df["H_flux"]) / ((flux_df["USTAR"] ** 3) * flux_df["theta"] * flux_df["rho_air"] * Cp)
        flux_df["energy_closure"] = (flux_df["LE_flux"] + flux_df["H_flux"]) / (flux_df["NETRAD"] - flux_df["G_flux"])
        if "G_flux" in flux_df.columns:
            flux_df["energy_closure"] = np.where(
                flux_df["energy_closure"].isna() & flux_df["G_flux"].isna(),
                (flux_df["LE_flux"] + flux_df["H_flux"]) / flux_df["NETRAD"],
                flux_df["energy_closure"]
            )
        else:
            flux_df["energy_closure"] = (flux_df["LE_flux"] + flux_df["H_flux"]) / flux_df["NETRAD"]

        # SOS/EOS
        sos_path = os.path.join(dir_SOS_EOS, f"{site_ID}_SOS_EOS.csv")
        sos_eos_df = pd.read_csv(sos_path).replace(-9999, np.nan)
        if not np.issubdtype(sos_eos_df["year"].dtype, np.integer):
            sos_eos_df["year"] = sos_eos_df["year"].astype("Int64")
        season_mask = pd.Series(False, index=flux_df.index)
        for _, row in sos_eos_df.dropna(subset=["year"]).iterrows():
            y = int(row["year"])
            w1 = w2 = None
            if pd.notna(row["SOS_1_GPP"]) and pd.notna(row["EOS_1_GPP"]):
                sos1, eos1 = int(row["SOS_1_GPP"]), int(row["EOS_1_GPP"])
                if sos1 <= eos1:
                    w1 = (flux_df["time_year"].eq(y) & flux_df["time_DOY"].between(sos1, eos1))
                else:
                    prev_part = (flux_df["time_year"].eq(y - 1) & (flux_df["time_DOY"] >= sos1))
                    curr_part = (flux_df["time_year"].eq(y) & (flux_df["time_DOY"] <= eos1))
                    w1 = prev_part | curr_part
            if pd.notna(row["SOS_2_GPP"]) and pd.notna(row["EOS_2_GPP"]):
                sos2, eos2 = int(row["SOS_2_GPP"]), int(row["EOS_2_GPP"])
                if sos2 <= eos2:
                    w2 = (flux_df["time_year"].eq(y) & flux_df["time_DOY"].between(sos2, eos2))
                else:
                    w2 = (flux_df["time_year"].eq(y) & ((flux_df["time_DOY"] >= sos2) | (flux_df["time_DOY"] <= eos2)))
            if (w1 is not None) and (w2 is not None):
                season_mask |= (w1 | w2)
            elif w1 is not None:
                season_mask |= w1
            elif w2 is not None:
                season_mask |= w2
        flux_df = flux_df[season_mask]

        # Site meta & cleanup
        for k, v in [("site_ID", site_ID), ("site_name", site_name), ("site_lat", site_lat),
                     ("site_lon", site_lon), ("site_elv", site_elv), ("IGBP", site_IGBP),
                     ("site_src", site_src)]:
            flux_df[k] = v
        flux_df.replace("", np.nan, inplace=True)
        flux_df.drop(columns=[c for c in drop_cols if c in flux_df.columns], inplace=True, errors="ignore")

        # atm selection
        if site_elv <= 500:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    ((flux_df["PA_F"] < 1020) & (flux_df["PA_F"] > 940))
                    )
        elif 500 < site_elv <= 1000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 870)
                    )
        elif 1000 < site_elv <= 2000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 770)
                    )
        elif 2000 < site_elv <= 3000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 670)
                    )
        else:  # site_elv > 3000
            day_atm_mask = (
                (flux_df["SW_IN_F"] >= 50) &
                ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                (flux_df["LE_flux"] >= 0) &
                (flux_df["TA_F"] >= 5)
            )
        flux_df_atm_sel = flux_df[day_atm_mask]
        count_valid = int(len(flux_df_atm_sel))

        # neg H & 2-hr runs
        neg = flux_df_atm_sel["H_flux"] < 0
        count_neg_H_tmp = int(neg.sum())
        delta = pd.Timedelta(minutes=30) if len_HH == 1 else pd.Timedelta(hours=1)
        times = flux_df_atm_sel["TIMESTAMP_END"]
        step_ok = times.diff().eq(delta) & neg
        grp = (~step_ok).cumsum()
        run_sizes = grp.map(grp.value_counts())
        need = 4 if len_HH == 1 else 2
        neg_H_df_sel_consec2hr = flux_df_atm_sel[neg & (run_sizes >= need)]
        count_neg_H_consec2hr_tmp = int((run_sizes[neg] >= need).sum())
        neg_H_df_sel = flux_df_atm_sel[neg]

        print(
            "finish searching site = ", site_ID,
            ", valid =", count_valid,
            ", number of negative H =", count_neg_H_tmp,
            ", number of 2hr negative H =", count_neg_H_consec2hr_tmp
        )

        return (neg_H_df_sel, neg_H_df_sel_consec2hr, count_valid, count_neg_H_tmp, count_neg_H_consec2hr_tmp)

    elif site_src == "LBA-ECO":
        # --------------- LBA-ECO (Excel) ---------------
        LBAECO_df = pd.read_excel(os.path.join(xlsx_dir, pattern), skiprows=[1])
        LBAECO_df["combined"] = LBAECO_df["Year_LBAMIP"].astype(int) * 1000 + LBAECO_df["DoY_LBAMIP"].astype(int)
        LBAECO_df["date"] = pd.to_datetime(LBAECO_df["combined"], format="%Y%j")
        LBAECO_df["datetime"] = LBAECO_df["date"] + pd.to_timedelta(LBAECO_df["Hour_LBAMIP"], unit="h")

        flux_df = pd.DataFrame(index=np.arange(len(LBAECO_df)))
        flux_df["site_ID"]   = site_ID
        flux_df["site_name"] = site_name
        flux_df["site_lat"]  = site_lat
        flux_df["site_lon"]  = site_lon
        flux_df["site_elv"]  = site_elv
        flux_df["IGBP"]      = site_IGBP
        flux_df["site_src"]  = site_src

        flux_df["TIMESTAMP_END"] = LBAECO_df["datetime"]
        flux_df["P_F"]           = LBAECO_df["prec"]
        flux_df["LE_flux"]       = LBAECO_df["LE"]
        flux_df["H_flux"]        = LBAECO_df["H"]
        flux_df["TA_F"]          = LBAECO_df["ta"]
        flux_df["PA_F"]          = LBAECO_df["press"]
        flux_df["GPP_NT_VUT_REF"]= LBAECO_df["GEP_model"]
        flux_df["GPP"]           = LBAECO_df["GEP_model"]
        flux_df["VPD_F"]         = LBAECO_df["VPD"]
        flux_df["SW_IN_F"]       = LBAECO_df["rgs"]
        flux_df["WS_F"]          = LBAECO_df["ws"]
        flux_df["LW_IN_F"]       = LBAECO_df["rgl"]
        flux_df["NETRAD"]        = LBAECO_df["Rn"]
        flux_df["G_flux"]        = LBAECO_df["FG"]
        flux_df["USTAR"]         = LBAECO_df["ust"]
        del LBAECO_df

        # units & timing
        flux_df["PA_F"]  = flux_df["PA_F"] * 10.0
        flux_df["VPD_F"] = flux_df["VPD_F"] * 10.0
        if (flux_df["TIMESTAMP_END"].iloc[1] - flux_df["TIMESTAMP_END"].iloc[0]) == pd.Timedelta(minutes=30):
            len_HH = 1
            flux_df["TIMESTAMP_START"] = flux_df["TIMESTAMP_END"] - pd.Timedelta(minutes=30)
        else:
            len_HH = 0
            flux_df["TIMESTAMP_START"] = flux_df["TIMESTAMP_END"] - pd.Timedelta(minutes=60)
        day_timestep = 48 if len_HH else 24
        rain_QC_timestep = 24 if len_HH else 12

        # QC1
        flux_df.replace(-9999, np.nan, inplace=True)

        # RH
        e_sat = es0 * np.exp((Lv / Rv) * ((1 / T0) - (1 / (flux_df["TA_F"] + 273.15))))
        flux_df["e_sat"] = e_sat
        flux_df["RH"] = 100.0 * (flux_df["e_sat"] - flux_df["VPD_F"]) / flux_df["e_sat"]
        flux_df.loc[~flux_df["RH"].between(0, 100), "RH"] = np.nan

        # Vectorized rainy-event screen
        flux_df["P_F"] = pd.to_numeric(flux_df["P_F"], errors="coerce")
        win = (48 if len_HH else 24) + 1
        rain_sum = flux_df["P_F"].rolling(win, min_periods=1).sum()
        rain_hit = rain_sum > 0
        cols_nan = ["USTAR","H_flux","LE_flux","GPP_NT_VUT_REF","GPP","G_flux"]
        cols_nan = [c for c in cols_nan if c in flux_df.columns]
        if cols_nan:
            flux_df.loc[rain_hit, cols_nan] = np.nan

        # Air density
        mixing_ratio = (epson * (flux_df["e_sat"] - flux_df["VPD_F"])) / (flux_df["PA_F"] - (flux_df["e_sat"] - flux_df["VPD_F"]))
        q = mixing_ratio / (1 + mixing_ratio)
        flux_df["rho_air"] = flux_df["PA_F"] * 100.0 / (Rd * (1 + 0.608 * q) * (flux_df["TA_F"] + 273.15))

        # USTAR bounds
        USTAR_mean = flux_df["USTAR"].mean(skipna=True); USTAR_std = flux_df["USTAR"].std(skipna=True)
        USTAR_upper = USTAR_mean + 2 * USTAR_std

        # QC3 outlier removal (simple set for LBA-ECO)
        flux_df["time_year"] = flux_df["TIMESTAMP_START"].dt.year
        flux_df["time_DOY"]  = flux_df["TIMESTAMP_START"].dt.dayofyear
        site_data = pd.DataFrame({"TIMESTAMP_START": flux_df["TIMESTAMP_START"], "TIMESTAMP_END": flux_df["TIMESTAMP_END"], "SW_IN_F": flux_df["SW_IN_F"]})
        vars_present = [v for v in var_to_QC3_LBAECO if v in flux_df.columns]
        for var in vars_present:
            df_var = flux_df[["TIMESTAMP_START","TIMESTAMP_END","SW_IN_F",var]].copy()
            df_no = remove_outlier(df_var, var)
            df_no.drop(columns=["hour_start"], inplace=True, errors="ignore")
            site_data = site_data.merge(df_no[["TIMESTAMP_START","TIMESTAMP_END",var]], on=["TIMESTAMP_START","TIMESTAMP_END"], how="left")

        # Replace QC3-cleaned targets
        KEYS = ["TIMESTAMP_START","TIMESTAMP_END"]
        targets = [c for c in var_to_QC3_LBAECO if c in site_data.columns]
        if targets:
            site_data = site_data.drop_duplicates(subset=KEYS, keep="last")
            f_idx = flux_df.set_index(KEYS); s_idx = site_data.set_index(KEYS)
            miss = [c for c in targets if c not in f_idx.columns]
            if miss: f_idx = f_idx.join(s_idx[miss], how="left")
            exist = [c for c in targets if c in f_idx.columns]
            if exist: f_idx.update(s_idx[exist], overwrite=True)
            flux_df = f_idx.reset_index()

        # Diagnostics
        flux_df.loc[~((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)), "USTAR"] = np.nan
        flux_df["theta"] = (flux_df["TA_F"] + 273.15) * ((1000.0 / flux_df["PA_F"]) ** (Rd / Cp))
        if np.isnan(z_measure):
            flux_df["z_L"] = np.nan
        else:
            flux_df["z_L"] = -(z_measure * vK_const * g * flux_df["H_flux"]) / ((flux_df["USTAR"] ** 3) * flux_df["theta"] * flux_df["rho_air"] * Cp)
        flux_df["energy_closure"] = (flux_df["LE_flux"] + flux_df["H_flux"]) / (flux_df["NETRAD"] - flux_df["G_flux"])
        if "G_flux" in flux_df.columns:
            flux_df["energy_closure"] = np.where(
                flux_df["energy_closure"].isna() & flux_df["G_flux"].isna(),
                (flux_df["LE_flux"] + flux_df["H_flux"]) / flux_df["NETRAD"],
                flux_df["energy_closure"]
            )
        else:
            flux_df["energy_closure"] = (flux_df["LE_flux"] + flux_df["H_flux"]) / flux_df["NETRAD"]

        # SOS/EOS
        sos_path = os.path.join(dir_SOS_EOS, f"{site_ID}_SOS_EOS.csv")
        sos_eos_df = pd.read_csv(sos_path).replace(-9999, np.nan)
        if not np.issubdtype(sos_eos_df["year"].dtype, np.integer):
            sos_eos_df["year"] = sos_eos_df["year"].astype("Int64")
        season_mask = pd.Series(False, index=flux_df.index)
        for _, row in sos_eos_df.dropna(subset=["year"]).iterrows():
            y = int(row["year"])
            w1 = w2 = None
            if pd.notna(row["SOS_1_GPP"]) and pd.notna(row["EOS_1_GPP"]):
                sos1, eos1 = int(row["SOS_1_GPP"]), int(row["EOS_1_GPP"])
                if sos1 <= eos1:
                    w1 = (flux_df["time_year"].eq(y) & flux_df["time_DOY"].between(sos1, eos1))
                else:
                    prev_part = (flux_df["time_year"].eq(y - 1) & (flux_df["time_DOY"] >= sos1))
                    curr_part = (flux_df["time_year"].eq(y) & (flux_df["time_DOY"] <= eos1))
                    w1 = prev_part | curr_part
            if pd.notna(row["SOS_2_GPP"]) and pd.notna(row["EOS_2_GPP"]):
                sos2, eos2 = int(row["SOS_2_GPP"]), int(row["EOS_2_GPP"])
                if sos2 <= eos2:
                    w2 = (flux_df["time_year"].eq(y) & flux_df["time_DOY"].between(sos2, eos2))
                else:
                    w2 = (flux_df["time_year"].eq(y) & ((flux_df["time_DOY"] >= sos2) | (flux_df["time_DOY"] <= eos2)))
            if (w1 is not None) and (w2 is not None):
                season_mask |= (w1 | w2)
            elif w1 is not None:
                season_mask |= w1
            elif w2 is not None:
                season_mask |= w2
        flux_df = flux_df[season_mask]

        # Site meta, cleanup
        flux_df["site_ID"] = site_ID
        flux_df["site_name"] = site_name
        flux_df["site_lat"] = site_lat
        flux_df["site_lon"] = site_lon
        flux_df["site_elv"] = site_elv
        flux_df["IGBP"] = site_IGBP
        flux_df["site_src"] = site_src
        flux_df.replace("", np.nan, inplace=True)
        flux_df.drop(columns=[c for c in drop_cols if c in flux_df.columns], inplace=True, errors="ignore")

        # atm selection
        if site_elv <= 500:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    ((flux_df["PA_F"] < 1020) & (flux_df["PA_F"] > 940))
                    )
        elif 500 < site_elv <= 1000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 870)
                    )
        elif 1000 < site_elv <= 2000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 770)
                    )
        elif 2000 < site_elv <= 3000:
            day_atm_mask = (
                    (flux_df["SW_IN_F"] >= 50) &
                    ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                    ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                    (flux_df["LE_flux"] >= 0) &
                    (flux_df["TA_F"] >= 5) &
                    (flux_df["PA_F"] > 670)
                    )
        else:  # site_elv > 3000
            day_atm_mask = (
                (flux_df["SW_IN_F"] >= 50) &
                ((flux_df["USTAR"] >= 0.2) & (flux_df["USTAR"] < USTAR_upper)) &
                ((flux_df["energy_closure"] >= 0.7) & (flux_df["energy_closure"] <= 1.3)) &
                (flux_df["LE_flux"] >= 0) &
                (flux_df["TA_F"] >= 5)
            )
        flux_df_atm_sel = flux_df[day_atm_mask]
        count_valid = int(len(flux_df_atm_sel))

        # neg H & 2-hr runs
        neg = flux_df_atm_sel["H_flux"] < 0
        count_neg_H_tmp = int(neg.sum())
        delta = pd.Timedelta(minutes=30) if len_HH == 1 else pd.Timedelta(hours=1)
        times = flux_df_atm_sel["TIMESTAMP_END"]
        step_ok = times.diff().eq(delta) & neg
        grp = (~step_ok).cumsum()
        run_sizes = grp.map(grp.value_counts())
        need = 4 if len_HH == 1 else 2
        neg_H_df_sel_consec2hr = flux_df_atm_sel[neg & (run_sizes >= need)]
        count_neg_H_consec2hr_tmp = int((run_sizes[neg] >= need).sum())
        neg_H_df_sel = flux_df_atm_sel[neg]

        print(
            "finish searching site = ", site_ID,
            ", valid =", count_valid,
            ", number of negative H =", count_neg_H_tmp,
            ", number of 2hr negative H =", count_neg_H_consec2hr_tmp
        )

        return (neg_H_df_sel, neg_H_df_sel_consec2hr, count_valid, count_neg_H_tmp, count_neg_H_consec2hr_tmp)

    else:
        print(f"[{site_ID}] Unsupported source.")
        return (pd.DataFrame(), pd.DataFrame(), 0, 0, 0)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Load site list and filter
    df_sitelist = pd.read_excel(os.path.join(dir_flux, fn_fluxlist))
    forest_site_mask = (
        (df_sitelist["IGBP"].isin(["EBF","ENF","DBF","DNF","MF"])) &
        ((df_sitelist["year_end"] - df_sitelist["year_start"] + 1) >= 3)
    )
    df_forest_sitelist = df_sitelist[forest_site_mask].reset_index(drop=True)
    total_site = len(df_forest_sitelist)
    print("total site =", total_site)

    # Build process args
    site_info_list = []
    for st in range(total_site):  # adjust range as needed, 152:LBA-ECO, BAN
        site_src = df_forest_sitelist.loc[st, "source"]
        site_ID = df_forest_sitelist.loc[st, "site_ID"]
        site_name = df_forest_sitelist.loc[st, "SITE_NAME"]
        site_IGBP = df_forest_sitelist.loc[st, "IGBP"]
        site_year_start = int(df_forest_sitelist.loc[st, "year_start"])
        site_year_end = int(df_forest_sitelist.loc[st, "year_end"])
        z_measure = df_forest_sitelist.loc[st, "measure_h"]
        site_lat = df_forest_sitelist.loc[st, "LOCATION_LAT"]
        site_lon = df_forest_sitelist.loc[st, "LOCATION_LONG"]
        site_elv = df_forest_sitelist.loc[st, "LOCATION_ELEV"]

        site_info_list.append((site_ID, site_name, site_year_start, site_year_end,
                               site_IGBP, site_src, z_measure, site_lat, site_lon, site_elv))

    print(f"Appended all {len(site_info_list)} sites!")
    site_id_to_igbp = {args[0]: args[4] for args in site_info_list}

    all_neg_H = []
    all_neg_H_consec2hr = []
    site_summary_rows = []

    # Multiprocessing
    from os import cpu_count
    max_workers = min(8, (cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_plot, args): args[0] for args in site_info_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing sites"):
            site_ID = futures[fut]
            try:
                neg_H_df, neg_H_df_consec2hr, count_valid, count_neg_H_tmp, count_neg_H_consec2hr_tmp = fut.result()
                if neg_H_df is not None and not neg_H_df.empty:
                    all_neg_H.append(neg_H_df)
                if neg_H_df_consec2hr is not None and not neg_H_df_consec2hr.empty:
                    all_neg_H_consec2hr.append(neg_H_df_consec2hr)
                site_summary_rows.append({
                    "site_ID": site_ID,
                    "site_IGBP": site_id_to_igbp.get(site_ID, None),
                    "count_valid": int(count_valid),
                    "count_neg_H": int(count_neg_H_tmp),
                    "count_neg_H_2hr": int(count_neg_H_consec2hr_tmp),
                })
            except Exception as e:
                print(f"Error processing site {site_ID}: {e}")

    # Save outputs
    if all_neg_H:
        try:
            df_neg_H = pd.concat(all_neg_H, ignore_index=True)
            df_neg_H.to_csv(os.path.join(out_dir, fn_neg_H), index=False)
            print(f"Saved {fn_neg_H}")
        except ValueError:
            print("No neg_H records to save (all empty frames).")
    else:
        print("No neg_H records to save.")

    if all_neg_H_consec2hr:
        try:
            df_neg_H_consec2hr = pd.concat(all_neg_H_consec2hr, ignore_index=True)
            df_neg_H_consec2hr.to_csv(os.path.join(out_dir, fn_neg_H_consec2hr), index=False)
            print(f"Saved {fn_neg_H_consec2hr}")
        except ValueError:
            print("No consecutive 2hr neg_H records to save (all empty frames).")
    else:
        print("No consecutive 2hr neg_H records to save.")

    if site_summary_rows:
        df_site_summary = pd.DataFrame(
            site_summary_rows,
            columns=["site_ID","site_IGBP","count_valid","count_neg_H","count_neg_H_2hr"]
        )
        df_site_summary.to_csv(os.path.join(out_dir, fn_site_summary), index=False)
        print(f"Saved {fn_site_summary}")
    else:
        print("No per-site summary rows to save.")

    print("All sites processed. Files saved (if any).")








