#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# cleanup_sims.py Cleans up your messy simulations
#
# Copyright 2019-2022 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
#
# cleanup_sims.py is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
# This package is distributed in the hope that it will be useful to other
# researches. IT DOES NOT COME WITH ANY WARRANTY WHATSOEVER; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# See <http://www.gnu.org/licenses/>.
################################################################################
"""# Cleanup_sims.py

Cleans up your messy simulations. Comes with its own XTC parser to skip
atomic coordinates, as we're only interested in timestamps.

Coverage and Unittest Report
----------------------------

Access the coverage report under:

https://kevinsawade.github.io/cleanup_sims/htmlcov/index.html

Access the unittest report under:

https://kevinsawade.github.io/cleanup_sims/htmlcov/html_report.html

"""


################################################################################
# Imports
################################################################################


from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
import argparse
import asyncio
import fcntl
import logging
import math
import os
import random
import re
import shutil
import string
import struct
import subprocess
import sys
import termios
from io import StringIO
from pathlib import Path
from typing import Callable, Generator, List, Literal, Optional, Union

import imohash
import MDAnalysis as mda
import MDAnalysis.transformations as trans
import numpy as np
from MDAnalysis.lib.formats.libmdaxdr import XTCFile

################################################################################
# Typing
################################################################################


PerFileTimestepPolicyType = Literal[
    "raise",
    "stop_iter_on_empty",
    "ignore",
    "compare_with_dt",
    "choose_next",
]


InterFileTimestepPolicyType = Literal[
    "raise",
    "ignore",
    "fix_conflicts",
]


FileExistsPolicyType = Literal[
    "raise",
    "overwrite",
    "continue",
    "check_and_continue",
    "check_and_overwrite",
]


################################################################################
# Globals
################################################################################


__all__ = ["cleanup_sims"]
_dryrun = True


################################################################################
# Util classes
################################################################################


class XTC:
    def __init__(self, xtc_file):
        self.filename = xtc_file

    def __enter__(self):
        self.file = XTCFile(str(self.filename), "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def n_atoms(self):
        frame = self.file.read()
        return len(frame.x)

    def __iter__(self):
        while True:
            try:
                frame = self.file.read()
                yield frame.step, frame.time
            except StopIteration:
                break


################################################################################
# Util functions
################################################################################


def update_gmx_environ(
    version: Optional[str] = "2020.6",
    cuda: Optional[bool] = True,
    AVX512: Optional[bool] = False,
) -> None: # pragma: no cover
    """Updates the current environment variables specified by a GMXRC.bash

    Keyword Args:
        version (str, optional): The gromacs version to use. You can provide
            any flavour of '2020.1', '2021.2', etc. Defaults to '2020.6'.
        cuda (bool, optional): Whether to use the cuda version.
            Defaults to True.
        AVX512 (bool, optional): Whether to use AVX512 vector extensions.
            Defaults to False.

    """
    print("Also disabling gmx quotes")
    gmx_disable_quotes()
    release = get_lsb()
    if not cuda and not AVX512:
        source_path = (
            f"/home/soft/gromacs/gromacs-{version}/inst/"
            "shared_{release}/bin/GMXRC.bash"
        )
    elif cuda and not AVX512:
        source_path = (
            f"/home/soft/gromacs/gromacs-{version}/inst/"
            f"cuda_shared_{release}/bin/GMXRC.bash"
        )
    elif not cuda and AVX512:
        raise Exception(
            "AVX512 True is only possible" "with cuda True at the same time."
        )
    else:
        source_path = (
            f"/home/soft/gromacs/gromacs-{version}/inst/"
            f"cuda_shared_AVX_512_{release}/bin/GMXRC.bash"
        )
    if not os.path.isfile(source_path):
        raise Exception(f"Could not find GMXRC.bash at {source_path}")
    print(f"sourcing {source_path} ...")
    shell_source(source_path)


def get_lsb() -> str:
    """Get the current lsb of Ubuntu systems.

    Returns:
        str: The current lsb. Most often '18.04', or '20.04'.

    """
    with open("/etc/lsb-release", "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if "DISTRIB_RELEASE" in line:
            return line.split("=")[-1]
    else:
        raise Exception(
            "Could not determine LSB release." " Maybe you're not using Ubuntu?"
        )


def shell_source(script: str): # pragma: no cover
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it.

    """
    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    _ = pipe.communicate()[0]
    _ = [line.decode() for line in _.splitlines()]
    output = []
    for i in _:
        if "eval" in i:
            break
        output.append(i)
    try:
        env = dict((line.split("=", 1) for line in output))
    except ValueError as e:
        number = int(re.search("#\d*", str(e)).group().replace("#", ""))
        print(output[number])
        print(output[number - 1])
        raise
    os.environ.update(env)


def gmx_disable_quotes() -> None:
    """Sets a gmx environment variable. True = Quotes, False = No Quotes."""
    os.environ["GMX_NO_QUOTES"] = "1"


ORDINAL = lambda n: "%d%s" % (
    n,
    "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
)


def _sort_files_by_part_and_copy(file: Union[Path, str]) -> int:
    file = Path(file)
    if "#" in str(file):
        raise Exception("Can currenlty not sort parts and copies of files.")
    filename = str(file.name)
    if "part" not in filename:
        return 1
    else:
        return int(filename.split(".")[1][4:])


def parts_and_copies_generator(files: List[Union[str, Path]]) -> Generator:
    return sorted(files, key=_sort_files_by_part_and_copy)


def _get_start_end_in_dt(
    start_time: float, end_time: float, dt: int
) -> tuple[int, int]:
    if dt == -1:
        return start_time, end_time
    start = int(math.ceil((start_time) / float(dt))) * dt
    end = int(math.floor((end_time) / float(dt))) * dt
    return start, end


def map_in_and_out_files(
    directories: Union[List[str]],
    out_dir: Union[str, Path],
    x: str = "traj_comp.xtc",
    pbc: str = "nojump",
    deffnm: Optional[str] = None,
    trjcat: bool = True,
) -> dict[Path, dict[Path, Path]]:
    """Maps in and out files."""
    mapped_sims = {}

    # if deffnm is not None and traj_comp was
    # not manually redefined change x
    if deffnm is not None and x == "traj_comp.xtc":
        x = deffnm + ".xtc"
    base_filename = x.split(".")[0]

    # fill the dict
    for directory in directories:
        mapped_sims[Path(directory)] = {}
        if trjcat:
            # if trjcat put the "temporary files" into the parent dir.
            if "_comp" in base_filename:
                cat_base_filename = base_filename.replace("_comp", "")
            else:
                cat_base_filename = base_filename
            out_file = (
                Path(out_dir)
                / directory.split("/./")[1]
                / f"{cat_base_filename}_{pbc}.xtc"
            )
            out_dir_ = Path(directory).parent
            mapped_sims[Path(directory)]["trjcat"] = out_file
        else:
            mapped_sims[Path(directory)]["trjcat"] = False
            out_dir_ = Path(out_dir)
        mapped_sims[Path(directory)] = {}
        mapped_sims[Path(directory)]["trjcat"] = out_file
        files = Path(directory).glob(x.replace(".xtc", "*.xtc"))
        p = re.compile(x.rstrip(".xtc") + r"(.xtc|.part\d{4}.xtc)")
        files = filter(lambda x: p.search(str(x)) is not None, files)
        files = list(parts_and_copies_generator(files))

        # if only one sim, put that into output. No cat
        if len(files) == 1:
            mapped_sims[Path(directory)]["trjcat"] = False
            out_dir_ = Path(out_dir)
            file = files[0]
            if "/./" in str(directory):
                out_file = Path(out_dir_) / directory.split("/./")[1]
                out_file /= file.name.replace(base_filename, base_filename + f"_{pbc}")
            else:
                out_file = Path(out_dir_) / file.name.replace(
                    base_filename, base_filename + f"_{pbc}"
                )
            out_file = Path(str(out_file).replace("_comp", ""))
            mapped_sims[Path(directory)][file] = out_file

        # put the tmp sims into input, then cat
        else:
            for file in files:
                if "/./" in str(directory):
                    out_file = Path(out_dir_) / directory.split("/./")[1]
                    out_file /= file.name.replace(base_filename, base_filename + f"_{pbc}")
                else:
                    out_file = Path(out_dir_) / file.name.replace(
                        base_filename, base_filename + f"_{pbc}"
                    )
                mapped_sims[Path(directory)][file] = out_file
    return mapped_sims


def _get_logger(
    logfile: Path = Path("sim_cleanup.log"),
    loggerName: str = "SimCleanup",
    singular: bool = False,
    loglevel: int = logging.INFO,
) -> logging.Logger:
    if not singular:
        raise Exception
    logger = logging.getLogger(name=loggerName)
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    # console
    fmt = "%(name)s %(levelname)8s [%(asctime)s]: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(loglevel)
    logger.addHandler(console_handler)

    # file
    fmt = (
        '%(name)s %(levelname)8s [%(asctime)s] ["%(pathname)s:'
        '%(lineno)s", in %(funcName)s]: %(message)s'
    )
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


async def get_times_from_file(file: Path) -> np.ndarray:
    with XTC(file) as f:
        times = np.vstack([d for d in f])
    return times


def get_atoms_from_file(file: Path) -> int:
    with XTC(file) as f:
        n_atoms = f.n_atoms()
    return n_atoms


async def check_file_with_dataset(
    file: Path,
    out_file: Path,
    data: dict[Path, Union[str, int, bool, np.ndarray]],
    metadata: dict[Union[str, Path], str],
    n_atoms: int = -1,
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.Logger] = None,
) -> dict:
    if logger is None:
        logger = _get_logger()
    # data whe have the should be in metadata we have the is

    if file == "trjcat":
        return {}

    # for files that have been removed from data, as they ware one-timestamp-files
    if file not in data:
        return {}

    data = data[file]

    command = {
        "inp_file": file,
        "out_file": out_file,
        "dt": data["dt"],
        "b": data["start"],
        "e": data["end"],
        "run": True,
    }

    if not out_file.is_file():
        logger.info(f"The file {out_file} does not exist. I will create it.")
        return {file: command}

    if out_file.is_file():
        if file_exists_policy == "raise":
            raise Exception(
                f"File {out_file} already exists. "
                f"Due to the chosen `{file_exists_policy=}` "
                "I have raised an Exception."
            )
        elif file_exists_policy == "overwrite":
            logger.debug(f"Will overwrite {out_file} without checking.")
            return {file: command}
        elif file_exists_policy == "continue":
            logger.debug(
                f"File {out_file} already exists. Continuing. This can "
                f"lead to problems later on."
            )
            return {}
        elif file_exists_policy == "check_and_overwrite":
            logger.debug(f"Will overwrite {out_file} when checks fail.")
        elif file_exists_policy == "check_and_continue":
            logger.debug(f"Will check {out_file} and raise, when checks fail.")
        else:
            raise Exception(f"Unkown file_exists_policy: {file_exists_policy}")

    # first check the atoms of the output file
    n_atoms_in_file = get_atoms_from_file(out_file)
    n_atoms_ok = True if n_atoms == -1 else n_atoms_in_file == n_atoms
    if n_atoms_ok:
        logger.debug(
            f"The number of atoms in the output file {out_file} is "
            f"correct ({n_atoms})."
        )
    else:
        logger.info(
            f"The number of atoms in the output file {out_file} is "
            f"incorrect ({n_atoms} was requested, {n_atoms_in_file=}). "
            f"I will still continue to check the times, maybe I am "
            f"just overwriting this file."
        )

    # check whether the output file has the correct times specified by the data
    times_in_output = metadata[out_file][:, 1].astype(int)
    times_should_be = np.arange(data["start"], data["end"] + 1, data["dt"])
    time_ok = np.array_equal(times_in_output, times_should_be)
    timesteps_in_output = np.unique(times_in_output[1:] - times_in_output[:-1])

    if not time_ok:
        logger.info(
            f"The times (start: {times_in_output[0]}, end: "
            f"{times_in_output[-1]}, dt: {timesteps_in_output}) in "
            f"the output file {out_file} are different, than "
            f"the requested times (start: {data['start']}, end: {data['end']}, "
            f"dt: {data['dt']}). Based on the chosen `file_exists_policy`, "
            f"I will overwrite or skip this file, or raise an Exception."
        )

    # make the output
    file_ok = time_ok and n_atoms_ok
    if file_ok:
        logger.info(f"The file {out_file} is ok. No need to change it.")
        return {}

    if not file_ok and file_exists_policy == "check_and_continue":
        raise Exception(
            f"The file {out_file} does not adhere to "
            f"the requested {data['start']=}, {data['end']=} "
            f"and {data['dt']=}, it has these characteristics: "
            f"(start: {times_in_output[0]}, end: {times_in_output[-1]}, "
            f"dt: {timesteps_in_output}). Also check the logs in "
            f"{logger.handlers[1].baseFilename}. "
            f"Set `file_exists_policy` to 'overwrite' or "
            f"'check_and_overwrite' to overwrite this file."
        )

    logger.info(
        f"I will overwrite the file {out_file} which has the wrong times "
        f"start: {times_in_output[0]} ps, end: {times_in_output[1]} ps, "
        f"dt: {timesteps_in_output} ps, with the correct times: "
        f"start: {data['start']} ps, end: {data['end']} ps and "
        f"dt: {data['dt']} ps."
    )
    return {file: command}


def feasibility_check(
    metadata: dict[str, np.ndarray],
    input_files: list[Path],
    dt: int = -1,
    max_time: int = -1,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Checks whether the input files and the dt and max time are feasible.

    Args:
        ds (xr.Dataset): The dataset generated in the `write_and_check_times`
            function.
        input_files (list[str]): The files to check, as the dataset contains
            input, output (and sometimes the final trjcat file).
        dt (int): The timestep requested.
        max_time (int): The max_time requested.

    Returns:
        bool: Wheter possible or not.

    """
    if logger is None:
        logger = _get_logger()
    if dt == -1 and max_time == -1:
        return True

    # all timestamps in the input data
    input_times = np.hstack([v[:, 1] for k, v in metadata.items() if k in input_files])

    # check the maxtime
    if max_time != -1:
        max_time_files = np.max(input_times)
        if max_time_files < max_time:
            logger.warning(
                f"The simulation at {Path(input_files[0]).parent} can't "
                f"be used with a {max_time=}, because the max time all "
                f"xtc files reach only goes up to {max_time_files=}."
            )
            return False

    # number of dt timesteps
    if dt != -1:
        if max_time == -1:
            max_time = np.max(input_times)
        n_timesteps = math.ceil(max_time / dt) + 1
        check_timestamps = input_times[input_times <= max_time]
        n_timesteps_in_files = (np.unique(check_timestamps) % dt == 0).sum()
        if n_timesteps != n_timesteps_in_files:
            logger.warning(
                f"The simulation at {Path(input_files[0]).parent} can't "
                f"be used with a {dt=}, because the number of timesteps "
                f"with a max_time of {max_time=} needs to be {n_timesteps=}, "
                f"but the files allow for {n_timesteps_in_files=}. These timesteps "
                f"are the result of these timestapms:\n\n{check_timestamps}\n\n"
                f"{(np.unique(input_times) % dt == 0)=}"
            )
            return False

    logger.info(
        f"The files in {input_files[0].parent} ({[f.name for f in input_files]})"
        f"can be used with a {max_time=} and a {dt=}."
    )

    return True


def get_start_end_in_dt(
    metadata: dict[Union[Path, str], np.ndarray],
    input_files: list[Path],
    dt: int = -1,
    max_time: int = -1,
    logger: Optional[logging.Logger] = None,
) -> dict[Path, Union[str, int, bool, np.ndarray]]:
    if logger is None:
        logger = _get_logger()
    out = {}

    for i, file in enumerate(input_files):
        out[file] = {
            "start": None,
            "end": None,
            "check": False,
            "run": False,
            "dt": dt,
            "times": None,
        }
        times = metadata[file][:, 1]
        start, end = times[[0, -1]]
        if max_time != -1:
            if end > max_time:
                logger.debug(
                    f"File {file=} ends at {end=} ps, which is larger than the "
                    f"requested {max_time=} ps. I will truncate the file at {max_time}"
                )
                end = max_time
        start_dt, end_dt = _get_start_end_in_dt(start, end, dt)
        if i != 0:
            previous_file = input_files[i - 1]
            previous_times = metadata[previous_file][:, 1]
            previous_start, previous_end = previous_times[[0, -1]]
            previous_start_dt = out[Path(previous_file)]["start"]
            previous_end_dt = out[Path(previous_file)]["end"]

            # check whether the file is a one timestamp file
            if len(times) == 1:
                logger.debug(
                    f"The file {file} is a single-timestep file. It is "
                    f"safe to discard such files in the algorithm."
                )
                out.pop(file)
                continue

            # compare with previous
            if end_dt < previous_end_dt and start_dt < previous_end_dt:
                logger.debug(
                    f"Comparing the files {previous_file=} and {file=} resulted in "
                    f"a discarding of the file {file=}. This file starts at {start=} ps"
                    f"and ends at {end=} pico seconds. The chosen times in mutliples "
                    f"of {dt=} are {start_dt=} and {end_dt=}. The end of this file is "
                    f"earlier than the end of the previous file ({previous_end=} ps) "
                    f"and thus this file carries no new frames and can be discarded."
                )
                out.pop(file)
                continue
            elif start_dt < previous_end_dt and end_dt > previous_end_dt:
                logger.debug(
                    f"Comparing the files {previous_file=} and {file=} resulted "
                    f"in an adjusted timestep. The file starts at {start=} ps"
                    f"and ends at {end=} pico seconds. The chosen times in mutliples "
                    f"of {dt=} are {start_dt=} and {end_dt=}. The start of this file is "
                    f"earlier than the end of the previous file ({previous_end=} ps) "
                    f"and thus I am adjusting the start_dt to {(previous_end_dt + dt)=}"
                )
                start_dt = previous_end_dt + dt
            else:
                logger.debug(f"Comparing the files {previous_file=} and {file=}")
                if start_dt - previous_end_dt != dt:
                    msg = (
                        f"Can't concatenate using the files {previous_file=} and {file=}. "
                        f"The time-step between these two files is {start=}-{previous_end=}="
                        f"{(start - previous_end)} ps, which does not correspond to the"
                        f"requested {dt=}. The last times of the previous file: "
                        f"{previous_times[-5:]=} and the first times in file: "
                        f"{times[:5]}. This is a reason to stop the algorithm here."
                    )
                    raise Exception(msg)
                else:
                    logger.debug(
                        f"Can concatenate the files {previous_file=} and "
                        f"{file=} with the requested {dt=}."
                    )

        # make sure the times are obtainable in the file
        should_be_times = np.arange(start_dt, end_dt + 1, dt)
        if not np.all(np.isin(should_be_times, times)):
            missing_timestamps = should_be_times[~np.isin(should_be_times, times)]
            msg = (
                f"The file {file=} can't be used with a start_time of {start_dt=} ps "
                f"and an end_time of {end_dt=} ps. This would require the timestamps "
                f"{should_be_times=} to all be present in the file. However, these "
                f"timestamps are not in the file: {missing_timestamps=}. You "
                f"can check the corresponding logs or use `gmx check` to see "
                f"what's wrong here."
            )
            raise Exception(msg)

        # write to the out-dict
        out[Path(file)] = {
            "start": start_dt,
            "end": end_dt,
            "check": True,
            "run": False,
            "dt": dt,
            "times": should_be_times,
        }

    return out


async def update_times_on_wrong_hash(
    metadata: dict[str : np.ndarray],
    metadata_file: Path,
    logger: Optional[logging.logger] = None,
) -> dict[str, np.ndarray]:
    """Updates the times on the files, that have changed hashes"""
    if logger is None:
        logger = _get_logger()
    changed_hashes = []
    keys = list(metadata.keys())
    for file in keys:
        if file == "file_hashes":
            continue
        timedata = metadata[file]
        old_hash = metadata["file_hashes"][metadata["file_hashes"][:, 0] == file, 1]
        assert len(old_hash) == 1, print(file, old_hash)
        old_hash = old_hash[0]
        try:
            new_hash = imohash.hashfile(file, hexdigest=True)
            if old_hash != new_hash:
                changed_hashes.append(file)
                logger.debug(
                    f"Since last checking, the file {file}, the hash changed "
                    f"from {old_hash} to {new_hash}. I will continue to check "
                    f"the times of that file."
                )
        except FileNotFoundError:
            logger.info(f"Since last checking the file {file} was deleted.")
            metadata.pop(file)
            metadata["file_hashes"] = metadata["file_hashes"][
                metadata["file_hashes"][:, 0] != file
            ]

    if not changed_hashes:
        logger.debug(
            f"Since last checking this simulation, no new files have been changed."
        )
        return metadata

    for file in changed_hashes:
        logger.info(
            f"Since last opening the metadata.npz for the file {file}, "
            f"the file has changed. Loading new times from that file."
        )
        times = await get_times_from_file(Path(file))
        # update times
        metadata[file] = times
        # and hash
        metadata["file_hashes"][metadata["file_hashes"][:, 0] == file, 1] = new_hash

    save_metadata(metadata, metadata_file)
    logger.debug(
        f"Saved new metadata.npz at {metadata_file}, because the files "
        f"{changed_hashes} have changed hashes."
    )
    return metadata


async def update_files_in_metdata(
    metadata: dict[str, np.ndarray],
    metadata_file: Path,
    files: dict[Union[str, Path], Path],
    logger: Optional[logging.Logger] = None,
) -> dict[str, np.ndarray]:
    if logger is None:
        logger = _get_logger()
    new_files = []
    for k, v in files.items():
        if k == "trjcat":
            continue
        if k.is_file():
            if k not in metadata:
                new_files.append(k)
        if v.is_file():
            if v not in metadata:
                new_files.append(v)
    if files["trjcat"]:
        if files["trjcat"].is_file():
            if files["trjcat"] not in metadata:
                new_files.append(files["trjcat"])

    if not new_files:
        logger.debug(
            f"Since last checking this simulation, no new files have been added."
        )
        return metadata

    for new_file in new_files:
        logger.info(
            f"Since last checking, the file {new_file} was added "
            f"to the simulation. Adding its metadata to the sims "
            f"metadata.npz."
        )
        timedata = await get_times_from_file(new_file)
        new_hash = imohash.hashfile(new_file, hexdigest=True)
        metadata[new_file] = timedata
        if str(new_file) in metadata["file_hashes"][:, 0]:
            metadata["file_hashes"][metadata["file_hashes"][:, 0] == file, 1] = new_hash
        else:
            metadata["file_hashes"] = np.vstack(
                [metadata["file_hashes"], [[str(new_file), new_hash]]]
            )

    save_metadata(metadata, metadata_file)
    return metadata


def save_metadata(
    metadata: dict[Union[Path, str], np.ndarray],
    metadata_file: Path,
) -> None:
    # make all keys str
    metadata = {str(k): v for k, v in metadata.items()}
    file_hashes = metadata["file_hashes"]
    file_hashes = np.array([[str(file), file_hash] for file, file_hash in file_hashes])
    metadata["file_hashes"] = file_hashes
    try:
        np.savez(metadata_file, **metadata)
    except Exception as e:
        raise Exception(f"Could not create metadata.npz at {metadata_file}") from e


def load_metadata(metadata_file: Path) -> dict[Union[Path, str], np.ndarray]:
    """Loads a npz file and makes all str Paths, if applicable."""
    metadata = dict(np.load(metadata_file))
    file_hashes = metadata["file_hashes"]
    file_hashes = np.array([[Path(file), file_hash] for file, file_hash in file_hashes])
    metadata = {Path(k): v for k, v in metadata.items() if k != "file_hashes"} | {
        "file_hashes": file_hashes
    }
    return metadata


async def write_and_check_times(
    simulation: tuple[Path, dict[Path, Path]],
    max_time: int = -1,
    dt: int = -1,
    n_atoms: int = -1,
    per_file_timestep_policy: PerFileTimestepPolicyType = "raise",
    inter_file_timestep_policy: InterFileTimestepPolicyType = "raise",
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.Logger] = None,
) -> dict:
    sim_dir, sim_files = simulation
    if logger is None:
        logger = _get_logger()
    data_file = sim_dir / "metadata.npz"
    out = {sim_dir: {}}

    input_files = [k for k, v in sim_files.items() if k != "trjcat"]
    output_files = [v for k, v in sim_files.items() if k != "trjcat" and v.is_file()]
    all_out_file = False
    if sim_files["trjcat"]:
        if sim_files["trjcat"].is_file():
            all_out_file: Path = sim_files["trjcat"]

    # create the file if it does not exist
    if not data_file.is_file():
        logger.debug(
            f"The simulation {sim_dir} is missing its metadata.nc file. "
            f"I will scan the files and collect the timesteps."
        )

        # frames and times of input files
        logger.debug(f"Adding source files for {sim_dir}.")
        times = await asyncio.gather(*[get_times_from_file(s) for s in input_files])
        times = {k: v for k, v in zip(input_files, times)}

        # frames and times of maybe exisiting output files
        logger.debug(f"Adding destination files dor {sim_dir} if available.")
        _ = await asyncio.gather(*[get_times_from_file(s) for s in output_files])
        times |= {k: v for k, v in zip(output_files, _)}

        # frames and times of the trjcat file
        if all_out_file:
            logger.debug(f"Adding the tjcat file: {all_out_file}.")
            _ = await get_times_from_file(all_out_file)
            times |= {all_out_file: _}

        assert len(times) == len(input_files) + len(output_files) + (
            1 if all_out_file else 0
        )

        # same with file fashes
        logger.debug(f"Collecting file hashes for {sim_dir}.")
        file_hashes = []
        for inp_file, out_file in sim_files.items():
            if inp_file != "trjcat":
                file_hashes.append(
                    [inp_file, imohash.hashfile(inp_file, hexdigest=True)]
                )
            if out_file:
                if out_file.is_file():
                    file_hashes.append(
                        [out_file, imohash.hashfile(out_file, hexdigest=True)]
                    )
        file_hashes = np.array(file_hashes)

        # assert hat all file hashes are in keys
        assert len(file_hashes) == len(input_files) + len(output_files) + (
            1 if all_out_file else 0
        )

        logger.debug(f"Saving metadata.npz for {sim_dir}.")
        save_metadata(times | {"file_hashes": file_hashes}, data_file)
        assert data_file.is_file()

    # open the file if it exists
    if data_file.is_file():
        logger.debug(f"Opening metadata.nc for simulation {sim_dir}.")
        metadata = load_metadata(data_file)
        if "arr_0" in metadata:
            metadata["file_hashes"] = metadata.pop("arr_0")

    # update and check hashes
    metadata = await update_times_on_wrong_hash(metadata, data_file, logger)

    # update all files in metadata
    metadata = await update_files_in_metdata(metadata, data_file, sim_files, logger)

    # check the input files for feasibility
    if not feasibility_check(metadata, input_files, dt, max_time, logger):
        raise Exception(
            f"dt and max_time with the files in {sim_dir} not "
            f"possible. Check the logs in {logger.handlers[1].baseFilename} "
            f"for more info."
        )

    # if feasible decide on start and end times of the input files
    start_end_times = get_start_end_in_dt(metadata, input_files, dt, max_time, logger)

    # check the all out file and continue if everything is ok based on that we start
    # the commands dictionary, that will be passed to the main cleanup_sims function
    if all_out_file:
        # all_out_file is requested and it is a file
        max_time_file = np.max(metadata[all_out_file][:, 1])
        if max_time != -1:
            max_time_ok = max_time == max_time_file
        else:
            max_time_ = np.max(
                [np.max(t[:, 1]) for k, t in metadata if k != "file_hashes"]
            )
            max_time_ok = max_time_ == max_time_file
        if not max_time_ok:
            logger.info(
                f"The file file which will be produced by trjcat {all_out_file} "
                f"already exists, but has the wrong maximum time. Requested was "
                f"{max_time} ps, but the file has {max_time_file} ps."
            )
        else:
            logger.debug(
                f"The file file which will be produced by trjcat {all_out_file} "
                f"already exists and has the correct maximum time ({max_time_file} ps)."
            )

        timesteps_file = np.unique(
            metadata[all_out_file][1:, 1] - metadata[all_out_file][:-1, 1]
        )
        if dt != -1:
            if len(timesteps_file) != 1:
                logger.info(
                    f"The output file {all_out_file} has uneven timedeltas:"
                    f"{timesteps_file}."
                )
                dt_ok = False
            else:
                timesteps_file = timesteps_file[0]
                dt_ok = timesteps_file == dt
        else:
            dt_ok = len(timesteps_file) == 1

        if not dt_ok:
            logger.info(
                f"The file file which will be produced by trjcat {all_out_file} "
                f"already exists, but has the wrong timesteps. Requested was "
                f"{dt} ps, but the file has {timesteps_file} ps."
            )
        else:
            logger.debug(
                f"The file file which will be produced by trjcat {all_out_file} "
                f"already exists and has the correct timestep ({dt} ps)."
            )

        if max_time_ok and dt_ok:
            logger.debug(
                f"The file {sim_files['trjcat']} has the correct maximal time, "
                f"and the correct timesteps."
            )
            return {}
        else:
            if file_exists_policy == "raise":
                raise Exception(
                    f"The file {sim_files['trjcat']} does not adhere to "
                    f"the requested {max_time=} and {dt=}, but file_exists_policy"
                    f"is set to 'raise'. Set it to something else to prevent this "
                    f"error. Also check the logs at {logger.handlers[1].baseFilename} "
                    f"for more info."
                )
            elif file_exists_policy == "continue":
                logger.warning(
                    f"The file {sim_files['trjcat']} does not adhere to the "
                    f"requested {max_time=} and {dt=}, however, due to "
                    f"file_exists_policy being set to 'continue' I will "
                    f"not overwrite this file."
                )
                commands = {}
            elif file_exists_policy == "check_and_continue":
                raise Exception(
                    f"The file {sim_files['trjcat']} does not adhere to "
                    f"the requested {max_time=} and {dt=}, but file_exists_policy"
                    f"is set to 'check_and+continue'. Set it to something else to prevent this "
                    f"error. Also check the logs at {logger.handlers[1].baseFilename} "
                    f"for more info."
                )
            elif file_exists_policy == "check_and_overwrite":
                commands = {
                    "trjcat": {"files": [], "dt": dt, "out_file": sim_files["trjcat"]}
                }
            elif file_exists_policy == "continue":
                logger.warning(
                    f"Will skil the file {all_out_file}, although it might "
                    f"have wrong max_time and timesteps."
                )
                commands = {}
            elif file_exists_policy == "overwrite":
                commands = {
                    "trjcat": {"files": [], "dt": dt, "out_file": sim_files["trjcat"]}
                }
            else:
                raise Exception(f"Unkown {file_exists_policy=}")
    else:
        # file does not already exist. create it
        if sim_files["trjcat"]:
            commands = {
                "trjcat": {"files": [], "dt": dt, "out_file": sim_files["trjcat"]}
            }
        # file is not requested but the remaining files need to be cleaned.
        else:
            commands = {}

    # iterate over the files in start_end_times and compare them with the existing files
    result = await asyncio.gather(
        *[
            check_file_with_dataset(
                file,
                out_file,
                start_end_times,
                metadata,
                n_atoms,
                file_exists_policy,
                logger,
            )
            for file, out_file in sim_files.items()
        ]
    )
    for d in result:
        commands.update(d)

    # add the trjcat files (maybe some files are ok, maybe some are not,
    # check_file_with_dataset returns the
    # minimum amount of work
    if "trjcat" in commands:
        commands["trjcat"]["files"] = [v for k, v in sim_files.items() if k != "trjcat"]
        if not commands["trjcat"]["files"]:
            raise Exception(
                f"trjcat list empty. Input data created from "
                f"the values of the dict commands, here are "
                f"the keys of the first values {list(commands.values()).keys()}"
            )

    return commands


async def create_ndx_files(
    simulations: dict[Path, dict[Path, Path]],
    s: str = "topol.tpr",
    deffnm: Optional[str] = None,
    n_atoms: int = -1,
    ndx_add_group_stdin: str = "",
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.Logger] = None,
) -> None:
    await asyncio.gather(
        *[
            create_ndx_file(
                simulation,
                s,
                deffnm,
                n_atoms,
                ndx_add_group_stdin,
                file_exists_policy,
                logger,
            )
            for simulation in simulations.keys()
        ]
    )


async def create_ndx_file(
    simulation: Path,
    s: str = "topol.tpr",
    deffnm: Optional[str] = None,
    n_atoms: int = -1,
    ndx_add_group_stdin: str = "",
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.logger] = None,
) -> None:
    if deffnm is not None and s == "topol.tpr":
        s = deffnm + ".tpr"
    tpr_file = simulation / s
    ndx_file = simulation / "index.ndx"
    if logger is None:
        logger = _get_logger()
    overwrite = False
    check = False
    if ndx_file.is_file():
        if file_exists_policy == "raise":
            raise Exception(
                f"File {ndx_file} already exists. "
                f"Due to the chosen `{file_exists_policy=}` "
                "I have raised an Exception."
            )
        elif file_exists_policy == "overwrite":
            logger.debug(f"Will overwrite {ndx_file} without checking.")
            overwrite = True
        elif file_exists_policy == "continue":
            logger.debug(f"File {ndx_file} already exists. Continuing")
            return
        elif file_exists_policy == "check_and_overwrite":
            logger.debug(f"Will overwrite {ndx_file} when checks fail.")
            overwrite = True
            check = True
        elif file_exists_policy == "check_and_continue":
            logger.debug(f"Will check {ndx_file} and raise, when checks fail.")
            overwrite = False
            check = True
        else:
            raise Exception(f"Unkown file_exists_policy: {file_exists_policy=}")
    else:
        overwrite = True

    if check and n_atoms != -1:
        text = ndx_file.read_text()
        group_name = re.findall(r"\[(.*?)\]", text)[-1].strip()
        text = text.split("]")[-1].strip()
        fields = [row for line in text.splitlines() for row in line.split()]
        if len(fields) != n_atoms:
            if not overwrite:
                raise Exception(
                    f"{ndx_file} indexes the wrong number of atoms. "
                    f"{n_atoms} was requested, but the file contains "
                    f"{len(fields)} atoms for the new group: {group_name}. "
                    f"set `file_exists_policy` to 'overwrite' or "
                    f"'check_and_overwrite' to overwrite this file."
                )
            else:
                logger.debug(
                    f"{ndx_file} indexes the wrong number of atoms. "
                    f"{n_atoms} was requested, but the file contains "
                    f"{len(fields)} atoms for the new group: {group_name}. "
                    f"I will overwrite this file."
                )
        else:
            logger.debug(
                f"{ndx_file} is fine. The group {group_name} has the correct "
                f"number of atoms: {n_atoms}."
            )
            return

    if overwrite:
        ndx_file.unlink(missing_ok=True)

    cmd = f"gmx make_ndx -f {tpr_file} -o {ndx_file}"
    ndx_add_group_stdin += "\nq\n"
    ndx_add_group_stdin = ndx_add_group_stdin.encode()
    if not _dryrun:
        proc = await asyncio.subprocess.create_subprocess_shell(
            cmd=cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(ndx_add_group_stdin)
        if not ndx_file.is_file():
            print(proc.returncode)
            print(stderr.decode())
            print(stdout.decode())
            print(cmd)
            raise Exception(f"Could not create file {ndx_file}")
        else:
            print(f"Created {ndx_file}")
    else:
        logger.warning(f"DRY-RUN: Created ndx_file {ndx_file}")

    text = ndx_file.read_text()
    group_name = re.findall(r"\[(.*?)\]", text)[-1].strip()
    text = text.split("]")[-1].strip()
    fields = [row for line in text.splitlines() for row in line.split()]
    if len(fields) != n_atoms:
        raise Exception(
            f"{ndx_file} indexes the wrong number of atoms. "
            f"{n_atoms} was requested, but the file contains "
            f"{len(fields)} atoms for the new group: {group_name}. "
            f"You can try and provide a different `ndx_add_group_stdin`."
            f"Try to find the correct stdin by calling the command: "
            f"{cmd} manually."
        )


async def run_command_and_check(
    command: dict,
    logger: Optional[logging.Logger] = None,
) -> None:
    stdout = ""
    cmd = command["cmd"]
    stdin = command["stdin"].encode()
    out_file = command["out_file"]
    b = command["b"]
    e = command["e"]
    dt = command["dt"]

    # at this point we can be certain, that the out file is a bad file
    if out_file.is_file():
        logger.info(f"Deleting the trjcat file {out_file}.")
    if not _dryrun:
        out_file.unlink(missing_ok=True)
    else:
        logger.warning(f"DRY-RUN: Deleting {out_file}.")

    # making directories
    if not out_file.parent.is_dir():
        if not _dryrun:
            out_file.parent.mkdir(parents=True)
        else:
            logger.warning(f"DRY-RUN: Creating directory: {out_file.parent}.")

    logger.debug(f"Running command {cmd}.")
    if not _dryrun:
        print(cmd)
        proc = await asyncio.subprocess.create_subprocess_shell(
            cmd=cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(stdin)
        if not out_file.is_file():
            print(proc.returncode)
            print(stderr.decode())
            print(stdout.decode())
            print(cmd)
            raise Exception(f"Could not create file {out_file}")
        else:
            logger.debug(f"Created {out_file}")
    else:
        logger.warning(f"DRY-RUN: Created {out_file}")

    # run tests on the new file
    times = (await get_times_from_file(out_file))[:, 1]
    timesteps = np.unique(times[1:] - times[:-1])
    start_ok = times[0] == b
    end_ok = times[-1] == e
    if times[0] == times[-1]:
        if times[0] % dt == 0:
            timestep_ok = [dt]
        else:
            timestep_ok = []
    else:
        timestep_ok = timesteps

    if len(timestep_ok) == 1:
        timestep_ok = timestep_ok[0]
        timestep_ok = timestep_ok == dt
    else:
        timestep_ok = False

    file_ok = start_ok and end_ok and timestep_ok
    if file_ok:
        logger.info(
            f"The creation of the file {out_file} succeeded. All parameters are ok."
        )
        return stdout.decode()

    random_hash = "".join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        for _ in range(16)
    )
    stderr_file = Path(f"/tmp/{random_hash}.stderr")
    stdout_file = Path(f"/tmp/{random_hash}.stdout")
    with open(stderr_file, "w") as f:
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        f.write(stderr)
    with open(stdout_file, "w") as f:
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        f.write(stdout)

    logger.critical(
        f"Creation of the file {out_file} resulted in a wrong file. "
        f"The requested parameters were: start: {b} ps, end: {e} ps,"
        f"dt: {dt} ps. But the new file has these parameters: "
        f"start: {times[0]} ps, end: {times[-1]} ps, dt: {timesteps} ps. "
        f"The original command was: {cmd}. I do not know, why gromacs "
        f"does this. You can check the stdout and stderr of this process "
        f"using these files: {stderr_file}, {stdout_file}. I will now try "
        f"to attempt to use MDAnalysis as a fallback."
    )

    await mdanalysis_fallback(
        input_file=command["inp_file"],
        output_file=command["out_file"],
        tpr_file=command["s"],
        ndx_file=command["n"],
        b=command["b"],
        e=command["e"],
        dt=command["dt"],
        output_group_and_center=command["stdin"],
        n_atoms=command["n_atoms"],
        logger=logger,
    )

    # check the file again
    times = (await get_times_from_file(out_file))[:, 1]

    start_ok = times[0] == b
    end_ok = times[-1] == e
    timesteps = np.unique(times[1:] - times[:-1])
    timestep_ok = timesteps
    if len(timestep_ok) == 1:
        timestep_ok = timestep_ok[0]
        timestep_ok = timestep_ok == dt
    else:
        timestep_ok = False

    file_ok = start_ok and end_ok and timestep_ok
    if file_ok:
        logger.info(
            f"The creation of the file {out_file} succeeded with the "
            f"MDAnalysis method. All parameters are ok."
        )
        return

    logger.critical(
        f"Even the MDAnalysis fallback method to creat the file "
        f"{out_file} resulted in a wrong file. "
        f"The requested parameters were: start: {b} ps, end: {e} ps,"
        f"dt: {dt} ps. But the new file has these parameters: "
        f"start: {times[0]} ps, end: {times[-1]} ps, dt: {timesteps} ps. "
        f"The original command was: {cmd}. I do not know, why MDAnalysis "
        f"does this."
    )

    return stdout


async def mdanalysis_fallback(
    input_file: Path,
    output_file: Path,
    tpr_file: Path,
    ndx_file: Path,
    b: int,
    e: int,
    dt: int = -1,
    output_group_and_center: Optional[Union[str, int]] = None,
    n_atoms: int = -1,
    logger: Optional[logging.Logger] = None,
) -> None:
    # I have no idea why, but the local variable e gets deleted somewhere in the
    # function
    ee = e

    try:
        u = mda.Universe(str(tpr_file), str(input_file))
    except ValueError as e:
        n_atoms_tpr = int(str(e).splitlines()[1].split()[-1])
        n_atoms_xtc = int(str(e).splitlines()[2].split()[-1])

        if n_atoms_tpr == n_atoms_xtc:
            pass
        elif n_atoms_tpr > n_atoms_xtc and n_atoms_xtc == n_atoms:
            pass
        else:
            msg = (
                f"Can't use the MDAnalysis fallback with the requested "
                f"{n_atoms=}. .tpr file has {n_atoms_tpr} atoms, "
                f".xtc file has {n_atoms_xtc} atoms. I can't produce "
                f"a trajectory with consistent atoms with these files, as "
                f"I can't deduce which atoms are in the xtc and which in the "
                f"tpr. If both of them have the same number of atoms, that "
                f"would have been possible. It would also have been possible, "
                f"if the .xtc file has {n_atoms=} atoms and the .tpr file has "
                f"more atoms than the .xtc file, as I can use the .ndx file "
                f"to produce a .tpr with the correct number of atoms."
            )
            raise Exception(msg) from e

    # check whether the group indexes the correct number of atoms
    if "\n" in output_group_and_center:
        center = True
        group = output_group_and_center.split("\n")[0]
    else:
        center = False
        group = output_group_and_center
    ndx_content = ndx_file.read_text()
    lines = ndx_content.split(f"{group} ]")[1].split("[")[0]
    atoms = []
    for line in lines.splitlines():
        atoms.extend(list(map(int, line.split())))

    # raise exception if bad
    if len(atoms) != n_atoms:
        raise Exception(
            f"The group {group} does not index the correct number of "
            f"atoms. Requested was {n_atoms=}, but the group indexes "
            f"{len(atoms)=} atoms. I can't continue from here."
        )

    # create a new temporary tpr file for MDAnalysis
    random_hash = "".join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        for _ in range(16)
    )
    tmp_tpr = Path(f"/tmp/{random_hash}.tpr")
    cmd = f"gmx convert-tpr -s {tpr_file} -o {tmp_tpr} -n {ndx_file}"
    if not _dryrun:
        proc = await asyncio.subprocess.create_subprocess_shell(
            cmd=cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=(group + "\n").encode())
        if not tmp_tpr.is_file():
            print(proc.returncode)
            print(stderr.decode())
            print(stdout.decode())
            print(cmd)
            raise Exception(
                f"Could not create file {tmp_tpr} which is bad, because "
                f"gromacs did not produce the correct file, and I am "
                f"already at the fallback procedure using MDAnalysis."
            )
        else:
            logger.debug(f"created {tmp_tpr}")
    else:
        logger.warning(f"DRY-RUN: Created {tmp_tpr}.")

    # load
    u = mda.Universe(str(tmp_tpr), str(input_file))
    ag = u.select_atoms("all")

    # add transformations
    transformations = [trans.unwrap(ag)]
    if center:
        transformations.extend([trans.center_in_box(ag, wrap=True), trans.wrap(ag)])
    u.trajectory.add_transformations(*transformations)

    # define timestamps
    timestamps = np.arange(b, ee + 1, dt)

    # write the timesteps
    if not _dryrun:
        with mda.Writer(str(output_file), ag.n_atoms) as w:
            for ts in u.trajectory:
                if ts.time in timestamps:
                    w.write(ag)
    else:
        logger.warning(f"DRY-RUN: Created {output_file} with MDAnalysis fallback.")


async def run_async_commands(
    commands: list[dict],
    logger: Optional[logging.Logger] = None,
) -> None:
    return await asyncio.gather(*[run_command_and_check(c, logger) for c in commands])


async def prepare_sim_cleanup(
    simulations: dict[Path, dict[Path, Path]],
    max_time: int = -1,
    dt: int = -1,
    n_atoms: int = -1,
    per_file_timestep_policy: PerFileTimestepPolicyType = "raise",
    inter_file_timestep_policy: InterFileTimestepPolicyType = "raise",
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.logger] = None,
) -> dict:
    plan = await asyncio.gather(
        *[
            write_and_check_times(
                simulation,
                max_time,
                dt,
                n_atoms,
                per_file_timestep_policy,
                inter_file_timestep_policy,
                file_exists_policy,
                logger,
            )
            for simulation in simulations.items()
        ]
    )
    return plan


async def async_run_concat_commands(
    commands: list[dict[str, Union[str, Path, int]]],
    logger: Optional[logging.Logger] = None,
) -> None:
    await asyncio.gather(
        *[run_concat_command(command, logger=logger) for command in commands]
    )


async def run_concat_command(
    command: dict[str, Union[str, Path, int]],
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = _get_logger()
    b = command["b"]
    e = command["e"]
    dt = command["dt"]
    out_file = command["out_file"]

    # create the cat files
    cat_files = []
    for file in command["inp_files"]:
        if file.is_file():
            cat_files.append(file)
        else:
            logger.info(
                f"The output file {file} does not exist and will not "
                f"be provided to gmx trjconv. It is probably a file with "
                f"no timesteps (only a single frame)."
            )
    cat_files = " ".join(map(str, cat_files))
    command["cmd"] = command["cmd"].replace("CAT_FILES", cat_files)
    cmd = command["cmd"]

    # run the command
    # at this point we can be certain, that the out file is a bad file
    if out_file.is_file():
        logger.info(f"Deleting the trjcat file {out_file}.")
    if not _dryrun:
        out_file.unlink(missing_ok=True)
    else:
        logger.warning(f"DRY-RUN: Deleted {out_file}")

    # making directories
    if not out_file.parent.is_dir():
        if not _dryrun:
            out_file.parent.mkdir(parents=True)
        else:
            logger.warning(f"DRY-RUN: Creating directory: {out_file.parent}.")

    logger.debug(f"Running command {cmd}.")
    if not _dryrun:
        proc = await asyncio.subprocess.create_subprocess_shell(
            cmd=cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if not out_file.is_file():
            print(proc.returncode)
            print(stderr.decode())
            print(stdout.decode())
            print(cmd)
            raise Exception(f"Could not create file {out_file}")
        else:
            logger.debug(f"Created {out_file}")
    else:
        logger.warning(f"DRY-RUN: Created {out_file}")

    # run tests on the new file
    times = (await get_times_from_file(out_file))[:, 1]
    timesteps = np.unique(times[1:] - times[:-1])
    start_ok = times[0] == b
    end_ok = times[-1] == e or e == -1
    timestep_ok = np.unique(times[1:] - times[:-1])
    if len(timestep_ok) == 1:
        timestep_ok = timestep_ok[0]
        timestep_ok = timestep_ok == dt
    else:
        timestep_ok = False

    file_ok = start_ok and end_ok and timestep_ok
    if file_ok:
        logger.info(
            f"The creation of the file {out_file} succeeded. All parameters are ok."
        )
        return

    random_hash = "".join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        for _ in range(16)
    )
    stderr_file = Path(f"/tmp/{random_hash}.stderr")
    stdout_file = Path(f"/tmp/{random_hash}.stdout")
    with open(stderr_file, "w") as f:
        if isinstance(stderr, bytes):
            stderr = stderr.decode()
        f.write(stderr)
    with open(stdout_file, "w") as f:
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        f.write(stdout)

    logger.critical(
        f"Creation of the file {out_file} resulted in a wrong file. "
        f"The requested parameters were: start: {b} ps, end: {e} ps,"
        f"dt: {dt} ps. But the new file has these parameters: "
        f"start: {times[0]} ps, end: {times[-1]} ps, dt: {timesteps} ps. "
        f"The original command was: {cmd}. I do not know, why gromacs "
        f"does this. You can check the stdout and stderr of this process "
        f"using these files: {stderr_file}, {stdout_file}. I will now try "
        f"to attempt to use MDAnalysis as a fallback."
    )

    return


async def async_run_create_pdb(
    pdb_commands: list[dict[str, Union[str, Path, int]]],
    n_atoms: int = -1,
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.Logger] = None,
) -> None:
    await asyncio.gather(
        *[
            async_create_pdb(pdb_cmd, n_atoms, file_exists_policy, logger=logger)
            for pdb_cmd in pdb_commands
        ]
    )


async def async_create_pdb(
    pdb_cmd: dict[str, Union[str, Path, int]],
    n_atoms: int = -1,
    file_exists_policy: FileExistsPolicyType = "raise",
    logger: Optional[logging.Logger] = None,
) -> None:
    pdb_out_file = pdb_cmd["out_file"]
    check = False
    if pdb_out_file.is_file():
        if file_exists_policy == "raise":
            raise Exception(
                f"File {pdb_out_file} already exists. "
                f"Due to the chosen `{file_exists_policy=}` "
                "I have raised an Exception."
            )
        elif file_exists_policy == "overwrite":
            logger.debug(f"Will overwrite {pdb_out_file} without checking.")
            overwrite = True
        elif file_exists_policy == "continue":
            logger.debug(f"File {pdb_out_file} already exists. Continuing")
            return
        elif file_exists_policy == "check_and_overwrite":
            logger.debug(f"Will overwrite {pdb_out_file} when checks fail.")
            overwrite = True
            check = True
        elif file_exists_policy == "check_and_continue":
            logger.debug(f"Will check {pdb_out_file} and raise, when checks fail.")
            overwrite = False
            check = True
        else:
            raise Exception(f"Unkown file_exists_policy: {file_exists_policy=}")
    else:
        overwrite = True

    if check and n_atoms != -1:
        n_atoms_in_file = mda.Universe(str(pdb_out_file)).atoms.n_atoms
        if n_atoms_in_file != n_atoms:
            if not overwrite:
                raise Exception(
                    f"{pdb_out_file} has the wrong number of atoms. "
                    f"{n_atoms} was requested, but the file contains "
                    f"{n_atoms_in_file} atoms. "
                    f"Set `file_exists_policy` to 'overwrite' or "
                    f"'check_and_overwrite' to overwrite this file."
                )
            else:
                logger.debug(
                    f"{pdb_out_file} indexes the wrong number of atoms. "
                    f"{n_atoms} was requested, but the file contains "
                    f"{n_atoms_in_file} atoms. "
                    f"I will overwrite this file."
                )
        else:
            logger.debug(
                f"{pdb_out_file} is fine. It has the correct "
                f"number of atoms: {n_atoms}."
            )
            return

    if overwrite:
        if not _dryrun:
            pdb_out_file.unlink(missing_ok=True)
        else:
            logger.warning(f"DRY-RUN: Deleted {pdb_out_file}")

    if not _dryrun:
        proc = await asyncio.subprocess.create_subprocess_shell(
            cmd=pdb_cmd["cmd"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(pdb_cmd["stdin"].encode())
        if not pdb_out_file.is_file():
            print(proc.returncode)
            print(stderr.decode())
            print(stdout.decode())
            print(pdb_cmd["cmd"])
            raise Exception(f"Could not create file {pdb_out_file}")
        else:
            print(f"Created {pdb_out_file}")
    else:
        logger.warning(f"DRY-RUN: Created {pdb_out_file}")

    n_atoms_in_file = mda.Universe(str(pdb_out_file)).atoms.n_atoms
    if n_atoms != -1:
        if n_atoms_in_file != n_atoms:
            raise Exception(
                f"{pdb_out_file} has the wrong number of atoms. "
                f"{pdb_out_file} was requested, but the file contains "
                f"{n_atoms_in_file}. "
                f"You can try and provide a different `output_group_and_center` "
                f"(-ndx-group for the CLI)."
                f"Try to find the correct stdin by calling the command: "
                f"{pdb_cmd['cmd']} manually."
            )


################################################################################
# Main functions
################################################################################


def cleanup_sims(
    directories: List[str],
    out_dir: Union[str, Path],
    dt: int = -1,
    max_time: int = -1,
    n_atoms: int = -1,
    s: str = "topol.tpr",
    x: str = "traj_comp.xtc",
    pbc: Union[str, None] = "nojump",
    center: bool = False,
    output_group_and_center: Union[str, int] = 1,
    deffnm: Optional[str] = None,
    trjcat: bool = True,
    create_pdb: bool = True,
    create_ndx: bool = False,
    ndx_add_group_stdin: Optional[str] = None,
    per_file_timestep_policy: PerFileTimestepPolicyType = "raise",
    inter_file_timestep_policy: InterFileTimestepPolicyType = "raise",
    file_exists_policy: FileExistsPolicyType = "raise",
    clean_copies: bool = False,
    dry_run: bool = False,
    logfile: Optional[Path] = Path("sim_cleanup.log"),
    loglevel: int = logging.INFO,
) -> None:
    """Cleans up your messy simulations.
    
    Uses asyncio and logging to make the process fast
    and transparent. Main use case is: Remove solvent and
    change timestep.
    
    Features include:

    * Caching of simulation times using metadata.npz files, so consecutive calls
        to this function are accelerated.
    * Using native GROMACS command (trjconv, trjcat) for working with
        xtc files. But if GROMACS does not play along, MDAnalysis will be used.
    * Using a fast xdr reader to get only times (not coordinates) from input
        trajectories.
    * Keeping track of files on disk via imohash. If a file changes through some
        external commands, the times and timesteps will be reloaded.

    The most important arguments are the `directories`, `out_dir`, `n_atoms`, `dt`,
    `max_time`. Take some time and read how they are used in this script.

    Although the script can be called from command line, I recommend writing a small
    python script to call this function.

    ```python
    #!/usr/bin/env python
    from cleanup_sims import cleanup_sims

    inp_dirs = ["/path/to/to/dir/./sims1", "/path/to/./sims2"]
    out_dir = ["/path/to/clean/sims"]

    cleanup_sims(inp_dirs, out_dir, dt=50, max_time=500)
    ```
    
    Note:
        The `directories` argument can include truncation marks which will keep the
        directory structure up to that point. For example, if you provide
        ['/path/to/./sim_folder/production'] as `directories` and '/home/me/' as
        `out_dir`, the simulation file without solvent can be found in
        '/home/me/sim_folder/production/traj.xtc'.

    Args:
        directories (list[str]): The input directories. You can provide as many
            as you like. You can include truncation marks similar to rsync in
            the directory names. These truncation marks will be used to determine
            the folder structure in the output_directory. Let's say we provide
            ./cleanup_sims.py -d /path/to/./first/simulation/xtcs -d /path/to/some/other/./simulation -o /output/dir,
            the directories /output/dir/first/simulation/xtcs and
            /output/dir/simulation will be created.
        out_dir (Union[str, Path]): The output directory, if you use the
            `--trjcat` flag, only the final concatenated file will be written to
            the folders in the output directory. The cleaned partial xtc files
            will be put into the same directory they are now in. If --trjcat is
            not given, all *xtc files in the input directories will be put in
            the folders created in the output directory.
        dt (int): Similar to Gromacs' dt option. Only writes frames every frame
            mod dt == 0 picoseconds. This is usually done using subprocess calls
            to gmx trjconv. However, sometimes gromacs screws up and makes a
            dt 100 ps to some dt 92 and some dt 8 ps. If -1 is given all frames
            will be written to output. Defaults to -1.
        max_time (int): The maximum time in ps to write trajectories. If some of
            your simulations don't reach that time, an exception will be thrown.
            If -1 is provided, the maximum time per xtcs in a directoyr is used.
            Defaults to -1. If the --trjcat option is provided and the output
            file fits the -dt and -max flags, the simulation cleanup of that
            simulation is considered finished. So consecutive calls will only
            change the file if the parameters -dt, -max (and -n-atom) change.
        s (str): The .tpr files in the directories. Similar to gromacs' -s flag.
            Will overwrite the values set with -deffnm. So setting
            -deffnm production -s some_tpr_file.tpr will look for
            some_tpr_file.tpr in the simulation directories.
        x (str): The .xtc files in the directories. Similar to gromacs' -f flag
            (which is -x in mdrun). Will overwrite the values set with -deffnm.
            So setting -deffnm production -x my_traj.xtc will look for
            my_traj.xtc, my_traj.part0001.xtc, my_traj.part0002.xtc and so on in
            the simulation directories. Defaults to traj_comp.xtc.
        pbc (str): What to provide for the periodic boundary correction of
            trjconv. Is set to nojump (best for single molecules) per default.
            Can alsobe explicitly set to None, if you don't want any pbc correction.
            The pbc method will be used as the name of the output file.
            So -pbc nojump will produce traj_nojump.xtc in your outout directories
            (if -trjcat is set) or traj_comp_nojump.xtc,
            traj_comp_nojump.part0001.xtc and so on, if -trjcat is not set.
            If -deffnm or -x are set, the filenames of these will be used, so
            that in theory my_traj_file_nojump.xtc and
            my_traj_file_nojump.part0001.xtc are possible.
        center (bool): Similar to gromacs trjconv's -[no]center option. If center
            is provided the -ndx-group will be used both for centering and pbc
            removal. If you want different groups for centering and output,
            use the python function (e.g. by writing a short py
            Use the python function and provide a string with newline chara\
                cter (\\n) to use different groups for pbc and center.
        output_group_and_center (str): The string to provide for gmx trjconv to
            center and remove pbcs from. Can either be an integer (0 is most of
            the times the system, 1 is most of the times the protein) or a string
            like System, Protein, or a custom group read from the ndx file, which
            is created if -create-ndx is provided. In any case, if you provide
            -n-atoms, the algorithm will check the output and inform you, when
            it contains a different number of atoms. This will allow you to
            tweak your group selection or make sure, that Gromacs recognizes
            your protein in group 1 correctly.
        deffnm (Optional[str]): The default filename for the files in the -d
            input directories. If you run your mdrun simulations with -deffnm
            production, you should also provide production for this argument.
            If -s or -x are set, this will be overwritten.
        trjcat (bool): Whether to concatenate the trajectories from the input
            directories into one long (-max) trajectory. If -trjcat is set, the
            output directory will only contain one .xtc file. The outputs from
            gmx trjconv will be written into the input directories along with
            the inputxtc files.
        create_pdb (bool): When given, the output directories will also contain
            start.pdb files that are extracted from the first frame of the
            simulations. These can be used to load the clean trajectries into
            other tools.
        create_ndx (bool): If gromac's doesn't recognize your protein as such
            and the index group 1 (Protein) contains the wrong number of atoms,
            you can create index.ndx files in the input directories with this
            option. See the -ndx-group-in flag how to do so.
        ndx_add_group_stdin (str): If you have non-standard residues in your
            protein and they are not included in group 1 (protein) of the
            standard index, you can add a custom group using this flag. If you
            have two non-standard residues (LYQ and GLQ) you can create a new
            group from the protein and the residue indices by providing the
            string "Protein | GLQ | LYQ" (these are logical or). This will use
            gmx make_ndx and the simulations .tpr file to create an index.ndx file.
            The -ndx-group flag should then be "Protein_GLQ_LYQ". If you are not
            sure, what to provide he re, play around with your tpr files and
            make_ndx and then start this program with what you learned from there.
        per_file_timestep_policy (PerFileTimestepPolicyType): What to do if
            the timesteps in the input files are bad. Possibilities are:
                * 'raise': Raise an Exception, if the timesteps are bad.
                * 'ignore': Continue on your merry way and ignore the
                    problems that the future might hold.
                * 'compare_with_dt': If the `dt` argument is divisible
                    by the bad timesteps, we're lucky and can just continue.
                * 'choose_next': If bad timesteps are detected, just choose
                    the closest available and continue with dt timesteps
                    from there.
            Defaults to 'raise'.
        inter_file_timetep_policy (InterFileTimestepPolicyType): What to do
            if the timeteps between two files hint at a gap. Possibilities are:
                * 'raise': Raise an Exception, if the difference between end of
                    traj_comp.partXXXX.xtc and start of traj_comp.partXXX+1.xtc
                    is larger than `dt` or even smaller than 0. Which can happen.
                * 'ignore': Not recommended, but a possibility.
                * 'fix_conflicts': We can try to fix the conflicts, if the data
                    in the input xtc files is there for all timesteps.
            Defaults to 'raise'.
        file_exists_policy (FileExistsPolicyType): What to do, if a file
            already exists. Possibilities are:
                * 'raise': Raise an Exception (aka FileExistsError).
                * 'overwrite': No compromises. Overwrite it.
                * 'continue': Continue looping over input and output files,
                    skipping, if a file already exists, without checking
                    whether times and timesteps in the file are correct.
                * 'check_and_continue': Check before continuing the loop,
                    it might be, that the old file is also wrong.
                * 'check_and_overwrite': Check before continuing the loop,
                    but if the file is wrong, fall back to overwriting it.
            Defaults to 'raise'.
        clean_copies (bool): Gromacs will leave file copies (#traj_comp.xtc.1#)
            in the directories when output files are already there. Delete
            the copy files. Defaults to False.
        dry_run (bool): If dry-run is set to True, no files will be written or
            deleted.
        logfile (str): Where to place the logfile at. Defaults to sim_cleanup.log.
        loglevel (str): What level to log at. Defaults to WARNING.

    """
    pbcs_ = ["mol", "res", "atom", "nojump", "cluster", "whole", "None"]
    if not pbc in pbcs_:
        raise Exception(f"Please choose one of {pbcs_} for `pbc`")
    if pbc == "None":
        pbc = None

    # get the logger
    logger = _get_logger(logfile, singular=True, loglevel=loglevel)

    # set global dryrun
    global _dryrun
    _dryrun = dry_run

    # set the terminator
    logging.StreamHandler.terminator = "\n"

    # print a start
    logger.info("Started to clean up simulations.")

    # if center is given, we need to duplicate the output
    output_group_and_center = str(output_group_and_center)
    if center:
        output_group_and_center = (
            output_group_and_center + "\n" + output_group_and_center + "\n"
        )
        center = "center"
    else:
        output_group_and_center += "\n"
        center = "nocenter"

    # check the input policies
    if per_file_timestep_policy not in PerFileTimestepPolicyType.__args__:
        raise ValueError(
            f"The `per_file_timestep_policy` needs to be one of the following: "
            f"{PerFileTimestepPolicyType}, but you provided: {per_file_timestep_policy}"
        )
    if inter_file_timestep_policy not in InterFileTimestepPolicyType.__args__:
        raise ValueError(
            f"The `inter_file_timestep_policy` needs to be one of the following: "
            f"{InterFileTimestepPolicyType}, but you provided: {inter_file_timestep_policy}"
        )
    if file_exists_policy not in FileExistsPolicyType.__args__:
        raise ValueError(
            f"The `file_exists_policy` needs to be one of the following: "
            f"{FileExistsPolicyType}, but you provided: {file_exists_policy}"
        )

    # get the input and the predefined output
    simulations = map_in_and_out_files(directories, out_dir, x, pbc, deffnm, trjcat)

    # based on this we can already collect the commands to create pdb files
    if create_pdb:
        pdb_commands = []
        for sim_dir, simulation in simulations.items():
            if len(simulation) == 2:
                for inp_file, out_file in simulation.items():
                    if inp_file != "trjcat":
                        tpr_file = inp_file.parent / s
                        break
            else:
                for inp_file, out_file in simulation.items():
                    # this is agnostic to whether trjcat is True or False
                    # important is only, whether the directories are the same
                    if inp_file == "trjcat":
                        tpr_file = sim_dir / s
                        break
                    if inp_file.parent != out_file.parent:
                        tpr_file = inp_file.parent / s
                        break

            if deffnm is not None and s == "topol.tpr":
                s = deffnm + ".tpr"
            pdb_out_file = out_file.parent / "start.pdb"
            pdb_cmd = (
                f"gmx trjconv -f {out_file} -s {tpr_file} -o {pdb_out_file} -dump 0"
            )
            if center:
                pdb_cmd += " -center"
            if create_ndx:
                pdb_cmd += f" -n {tpr_file.parent / 'index.ndx'}"
            pdb_cmd = {
                "cmd": pdb_cmd,
                "stdin": output_group_and_center,
                "out_file": pdb_out_file,
                "n_atoms": n_atoms,
            }
            pdb_commands.append(pdb_cmd)

    assert len(simulations) == len(directories)
    logger.info(f"{len(simulations)} simulations will be cleaned up.")

    # write the ndx files
    if create_ndx:
        asyncio.run(
            create_ndx_files(
                simulations,
                s,
                deffnm,
                n_atoms,
                ndx_add_group_stdin,
                file_exists_policy,
                logger,
            )
        )

    # prepeare everything
    # this method filters out what actually needs to be done and whether it is doable
    # out comes a dictionary that can be passed to asyncio
    plans = asyncio.run(
        prepare_sim_cleanup(
            simulations,
            max_time,
            dt,
            n_atoms,
            per_file_timestep_policy,
            inter_file_timestep_policy,
            file_exists_policy,
            logger,
        )
    )
    # to the plans, we add the tpr, ndx, center, s, output_group_and_center
    async_commands = []
    concat_commands = []
    for i, (plan, (sim_dir, sim_files)) in enumerate(zip(plans, simulations.items())):
        # find the tpr file in the directory
        if deffnm is not None and s == "topol.tpr":
            s = deffnm + ".tpr"
        tpr_file = sim_dir / s
        assert tpr_file.is_file(), print(f".tpr file {tpr_file} does not exist.")

        if create_ndx:
            ndx_file = sim_dir / "index.ndx"
            assert ndx_file.is_file(), print(f".tpr file {tpr_file} does not exist.")

        for inp_file, command in plan.items():
            if inp_file == "trjcat":
                command = {
                    "cmd": f"gmx trjcat -f CAT_FILES -o {command['out_file']}",
                    "b": 0,
                    "e": max_time,
                    "dt": dt,
                    "out_file": command["out_file"],
                    "stdin": output_group_and_center,
                    "n_atoms": n_atoms,
                    "inp_files": command["files"],
                    "s": tpr_file,
                }
                concat_commands.append(command)
            else:
                command = {
                    "cmd": f"gmx trjconv -s {tpr_file} -f {inp_file} -o {command['out_file']} "
                    f"-{center} -b {command['b']} -e {command['e']} -dt {command['dt']}",
                    "b": command["b"],
                    "e": command["e"],
                    "dt": command["dt"],
                    "out_file": command["out_file"],
                    "stdin": output_group_and_center,
                    "n_atoms": n_atoms,
                    "inp_file": inp_file,
                    "s": tpr_file,
                }
                if pbc is not None:
                    command["cmd"] += f" -pbc {pbc}"
                if create_ndx:
                    command["cmd"] += f" -n {ndx_file}"
                    command["n"] = ndx_file
                async_commands.append(command)

    # set the event loop for the gathers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # run the commands asynchronously
    if async_commands:
        result = loop.run_until_complete(
            run_async_commands(
                async_commands, logger,
            )
        )
        print(result)

    if concat_commands:
        loop.run_until_complete(
            async_run_concat_commands(
                concat_commands, logger=logger,
            )
        )

    if create_pdb:
        loop.run_until_complete(
            async_run_create_pdb(
                pdb_commands, n_atoms, file_exists_policy, logger=logger,
            )
        )

    if clean_copies:
        pass
        # copy_files = list(list(sims.values())[0].parent.glob("#*"))
        # logger.debug(
        #     f"Deleting {len(copy_files)} copy files (filenames like this: {copy_files[0]})"
        # )
        # for f in copy_files:
        #     f.unlink()

    logger.info("All finished. Rejoice.")


################################################################################
# Argparse and make it a script
################################################################################

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = vars(parser.parse_args(argv))
    return cleanup_sims(**args)

if __name__ == "__main__":
    raise SystemExit(main())
