#!/usr/bin/env python3

import datetime as dt
from pathlib import Path
import argparse
import cftime

import numpy as np
import xarray as xr

from Tracking_Functions_free_mem import moaap

DEFAULT_VARLIST = [
    "PSL", "Z500", "PRECT", "FLUT", "U850", "V850", "T850",
    "uIVT", "vIVT", "Q850", "U200", "V200"
]

def time_bounds_noleap(year, month, window_months=4, dt_hours=1):
    start = cftime.DatetimeNoLeap(year, month, 1, 0)

    new_month = month + window_months
    new_year = year + (new_month - 1) // 12
    new_month = (new_month - 1) % 12 + 1
    stop = cftime.DatetimeNoLeap(new_year, new_month, 1, 0) - dt.timedelta(hours=dt_hours) 

    return start, stop


def build_filenames(data_dir, casename, year, varlist):
    fnames = [
        f"{data_dir}/{casename}/{casename}.cam.h6.{var}.latlon_0.25x0.25_0E.{year}010100-{year+1}010100.nc"
        for var in varlist
    ]

    missing = [f for f in fnames if not Path(f).is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    return fnames


def flut_to_brightness_temp(olr: xr.DataArray) -> xr.DataArray:
    """Convert OLR [W m-2] to equivalent brightness temperature [K]."""
    sigma = 5.67e-8
    a, b = 1.228, -1.106e-3

    tf = (olr / sigma) ** 0.25
    disc = a**2 + 4 * b * tf
    disc = disc.clip(min=0)

    return (-a + xr.ufuncs.sqrt(disc)) / (2 * b)


def load_cesm_vars(fnames, start, stop, varlist, hr_stride=1):

    # Decode CF time to cftime objects (needed for noleap calendars and time slicing)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(
        fnames,
        combine="by_coords",
        coords="minimal",
        data_vars="minimal",
        compat="override",
        parallel=True,
        decode_times=time_coder
    )

    # Fix longitude (0..360 -> -180..180) and sort
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")

    # Subset by the requested time range
    ds = ds[varlist].sel(time=slice(start, stop))

    # Change from hourly to 6 hourly or 3 hourly if needed
    if hr_stride != 1:
        ds = ds.coarsen(time=hr_stride, boundary="trim").mean()
        print(f"Averaging hourly into {hr_stride} before tracking")
    else:
        print("Tracking with original hourly data")

    # Unit and variable conversions
    if "Z500" in ds:  ds["Z500"]  = ds["Z500"] * 9.81
    if "PRECT" in ds: ds["PRECT"] = ds["PRECT"] * 3.6e6
    if "FLUT" in ds:  ds["FLUT"]  = flut_to_brightness_temp(ds["FLUT"])

    out = ds.load()
    ds.close()
    return out


def run_moaap_window(
    data_dir: str,
    casename: str,
    year: int,
    month: int,
    out_dir: str,
    varlist=None,
    dt_hours: int = 1,
    hr_stride: int = 1,
    window_months: int = 4,
):

    if varlist is None:
        varlist = DEFAULT_VARLIST

    # Ensure deterministic order, remove duplicates while preserving order
    seen = set()
    varlist = [v for v in varlist if not (v in seen or seen.add(v))]

    output_folder = Path(out_dir) / casename
    output_folder.mkdir(parents=True, exist_ok=True)
    output_folder = f"{output_folder}/"

    fnames = build_filenames(data_dir, casename, year, varlist)
    print("Load variables from these files:")
    print(*fnames, sep="\n")

    t_start, t_stop = time_bounds_noleap(year, month, window_months=window_months, dt_hours=dt_hours)

    ds = load_cesm_vars(fnames, t_start, t_stop, varlist, hr_stride=hr_stride)

    Time = ds.indexes["time"].to_datetimeindex(unsafe=True, time_unit="ns")
    print(f"Time subset: {Time[0]} - {Time[-1]}")

    Lon, Lat = np.meshgrid(ds.lon.values, ds.lat.values)
    Mask = np.ones_like(Lon)

    def v(ds, name):
        da = ds.get(name)          # returns None if missing
        return None if da is None else da.values

    moaap(
        Lon, Lat, Time, dt_hours * hr_stride, Mask,
        v850=v(ds, "V850"),
        u850=v(ds, "U850"),
        t850=v(ds, "T850"),
        slp=v(ds, "PSL"),
        ivte=v(ds, "uIVT"),
        ivtn=v(ds, "vIVT"),
        z500=v(ds, "Z500"),
        q850=v(ds, "Q850"),
        v200=v(ds, "V200"),
        u200=v(ds, "U200"),
        pr=v(ds, "PRECT"),
        tb=v(ds, "FLUT"),
        DataName=casename,
        OutputFolder=str(output_folder),
    )

    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run MOAAP on CESM CAM h6 lat/lon output.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--casename", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--months-per-job", type=int, default=4)
    p.add_argument("--dt-hrs-data", type=int, default=1)
    p.add_argument("--dt-stride-tracking", type=int, default=1)
    p.add_argument("--varlist", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    varlist = None
    if args.varlist.strip():
        varlist = [v.strip() for v in args.varlist.split(",") if v.strip()]

    return run_moaap_window(
        data_dir=args.data_dir,
        casename=args.casename,
        out_dir=args.out_dir,
        year=args.year,
        month=args.month,
        window_months=args.months_per_job,
        dt_hours=args.dt_hrs_data,
        hr_stride=args.dt_stride_tracking,
        varlist=varlist,
    )


if __name__ == "__main__":
    raise SystemExit(main())

# # For testing only
# main([
#   "--data-dir", "/glade/campaign/univ/utam0017/MOAAP",
#   "--casename", "b.e13.HF-TNST.rcp85.ne120_t12.1920-2100.010",
#   "--out-dir", "/glade/derecho/scratch/jiangzhu/moaap/",
#   "--year", "2099",
#   "--month", "1",
# ])