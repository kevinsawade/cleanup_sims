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
"""Cleans up your messy simulations. Comes with its own XTC parser to skip
atomic coordinates, as we're only interested in timestamps.

"""


################################################################################
# Imports
################################################################################


from __future__ import annotations
import shlex
from datetime import datetime
import logging
import xarray as xr
import numpy as np
from pathlib import Path
import collections
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCReader
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
from copy import deepcopy
import math
import re
import os
from subprocess import PIPE, Popen
from subprocess import run as sub_run
import subprocess
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import mdtraj as md
import imohash
from xdrlib import Unpacker
from mdtraj.formats.xtc import XTCTrajectoryFile
import asyncio


################################################################################
# Typing
################################################################################


from typing import Literal, Union, List, Optional, Callable


PerFileTimestepPolicyType = Literal["raise",
                                    "stop_iter_on_empty",
                                    "ignore",
                                    "compare_with_dt",
                                    "choose_next",
                                   ]


InterFileTimestepPolicyType = Literal["raise",
                                      "ignore",
                                      "fix_conflicts",
                                      ]


FileExistsPolicyType = Literal["raise",
                               "overwrite",
                               "continue",
                               "check_and_continue",
                               "check_and_overwrite",
                              ]


################################################################################
# Util classes
################################################################################


def strseek(stream,
            tag,
            bufsize=10000):
    """ Find every position in file where tag is found. """
    v = len(tag)
    x = stream.read(bufsize)
    n = 0
    while len(x) >= v:
        m = x.find(tag)
        if m > -1:
            # If we found the tag, the position is the total length
            # read plus the index at which the tag was found
            n += m
            yield n
            # Now we strip the part up to and including the tag
            x = x[m+v:]
            n += v
        elif len(x) > v:
            # If the tag was not found, strip the string, keeping only
            # the last v-1 characters, as these could combine with the
            # next part to be read in.
            # n is incremented with the number of characters taken from
            # the current string x (len(x)-v+1)
            n += len(x)-v+1
            x = x[1-v:]
        if len(x) <= v:
            x += stream.read(bufsize)


class XTC:

    def __init__(self, xtc_file):
        self.filename = xtc_file

    def __enter__(self):
        self.file = XTCFile(str(self.filename), 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __iter__(self):
        while True:
            try:
                frame = self.file.read()
                yield frame.time
            except StopIteration:
                break


################################################################################
# Util functions
################################################################################


def update_gmx_environ(version: Optional[str] = '2020.6',
                       cuda: Optional[bool] = True,
                       AVX512: Optional[bool] = False) -> None:
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
        source_path = (f"/home/soft/gromacs/gromacs-{version}/inst/"
                       "shared_{release}/bin/GMXRC.bash")
    elif cuda and not AVX512:
        source_path = (f"/home/soft/gromacs/gromacs-{version}/inst/"
                       f"cuda_shared_{release}/bin/GMXRC.bash")
    elif not cuda and AVX512:
        raise Exception("AVX512 True is only possible"
                        "with cuda True at the same time.")
    else:
        source_path = (f"/home/soft/gromacs/gromacs-{version}/inst/"
                       f"cuda_shared_AVX_512_{release}/bin/GMXRC.bash")
    if not os.path.isfile(source_path):
        raise Exception(f"Could not find GMXRC.bash at {source_path}")
    print(f"sourcing {source_path} ...")
    shell_source(source_path)


def get_lsb() -> str:
    """Get the current lsb of Ubuntu systems.

    Returns:
        str: The current lsb. Most often '18.04', or '20.04'.

    """
    with open('/etc/lsb-release', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        if 'DISTRIB_RELEASE' in line:
            return line.split('=')[-1]
    else:
        raise Exception("Could not determine LSB release."
                        " Maybe you're not using Ubuntu?")


def shell_source(script: str):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it.

    """
    pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    _ = pipe.communicate()[0]
    _ = [line.decode() for line in _.splitlines()]
    output = []
    for i in _:
        if 'eval' in i:
            break
        output.append(i)
    try:
        env = dict((line.split("=", 1) for line in output))
    except ValueError as e:
        number = int(re.search('#\d*', str(e)).group().replace('#', ''))
        print(output[number])
        print(output[number - 1])
        raise
    os.environ.update(env)


def gmx_disable_quotes() -> None:
    """Sets a gmx environment variable. True = Quotes, False = No Quotes."""
    os.environ['GMX_NO_QUOTES'] = '1'


ORDINAL = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


def _sort_files_by_part_and_copy(file: Union[Path, str]) -> int:
    file = Path(file)
    if '#' in str(file):
        raise Exception("Can currenlty not sort parts and copies of files.")
    filename = str(file.name)
    if 'part' not in filename:
        return 1
    else:
        return int(filename.split('.')[1][4:])


def parts_and_copies_generator(files: List[Union[str, Path]]) -> Generator:
    return sorted(files, key=_sort_files_by_part_and_copy)


def get_start_end_in_dt(start_time: float,
                        end_time: float,
                        dt: int
                       ) -> Tuple[int, int]:
    start = int(math.ceil((start_time) / float(dt))) * dt
    end = int(math.floor((end_time) / float(dt))) * dt
    return start, end
    
    
def get_times_from_file(xtc: str,
                       ) -> Tuple[float, float, np.ndarray[Any, np.dtype[float]]]:
    """Uses MDAnalysis to get time data of an xtc file:
    
    Args:
        xtc (str): The file to use.
        
    Returns:
        tuple: A tuple contaiming:
            float: The start time in ps.
            float: The end time in ps.
    
    """
    factors = {'ps': 1, 'ns': 1000}
    reader = mda.coordinates.XTC.XTCReader(str(xtc), refresh_offsets=True)
    if reader.units['time'] not in factors:
        raise Exception(f"For file {xtc}: {reader.units} not in factors")
    start_ps = reader.trajectory[0].time
    end_ps = reader.trajectory[-1].time
    times = np.array([ts.time for ts in reader.trajectory])
    return start_ps, end_ps, times
    
    
def map_in_and_out_files(directories: Union[List[str]],
                         out_dir: Union[str, Path],
                         x: str = 'traj_comp.xtc',
                         pbc: str = 'nojump',
                         deffnm: Optional[str] = None,
                         trjcat: bool = True,
                        ) -> dict[Path, dict[Path, Path]]:
    """Maps in and out files."""
    mapped_sims = {}
    
    # if deffnm is not None and traj_comp was
    # not manually redefined change x
    if deffnm is not None and x == 'traj_comp.xtc':
        x = deffnm + '.xtc'
    base_filename = x.split('.')[0]
    
    # fill the dict
    for directory in directories:
        if trjcat:
            out_file = Path(out_dir) / directory.split('/./')[1] / f"{base_filename}_{pbc}.xtc"
            out_dir = Path(directory).parent
        else:
            mapped_sims["trjcat"] = False
        mapped_sims[Path(directory)] = {}
        mapped_sims[Path(directory)]["trjcat"] = out_file
        files = Path(directory).glob(x.replace('.xtc', '*.xtc'))
        p = re.compile(x.rstrip('.xtc') + r"(.xtc|.part\d{4}.xtc)")
        files = filter(lambda x: p.search(str(x)) is not None, files)
        files = list(parts_and_copies_generator(files))
        for file in files:
            if '/./' in str(directory):
                out_file = Path(out_dir) / directory.split('/./')[1]
                out_file /= file.name.replace(base_filename, base_filename + f'_{pbc}')
            else:
                out_file = Path(out_dir) / file.name.replace(base_filename, base_filename + f'_{pbc}')
            mapped_sims[Path(directory)][file] = out_file
    return mapped_sims


def check_exisiting_file(file: Path,
                         dt_should: int,
                         end_time_should: int,
                         start_time_should: int = 0,
                         file_exists_policy: FileExistsPolicyType = "raise",
                         start_ps: Optional[float] = None,
                         end_ps: Optional[float] = None,
                         times: Optional[np.ndarray] = None,
                         ) -> str:
    logger = logging.getLogger("SimCleanup")
    if not file.is_file():
        logger.debug(f"The file {file} does not exist. I will create it")
        return "create"
    
    msg = (f"File {file} already exists.")
    logger.debug(msg)
    if file_exists_policy == 'raise':
        msg += (f" Due to the chosen `file_exists_policy`={file_exists_policy} "
                "I have raised an Exception.")
        raise Exception(msg)
        
    elif file_exists_policy == "continue":
        msg += (f" Continuing without checks.")
        return "continue"
    
    elif file_exists_policy == "check_and_continue" or file_exists_policy == "check_and_overwrite":
        logger.debug(f"Checking timesteps in {file}.")
        if start_ps is None:
            start_ps, end_ps, times = get_times_from_file(str(file))
        
        # check the start_ps
        if start_ps != start_time_should:
            msg += (f" The start time of this file {start_ps} is different than the "
                   f"requested {start_time_should=}.")
            if file_exists_policy == 'check_and_overwrite':
                msg += " I will overwrite this file."
                logger.debug(msg)
                return "create"
            else:
                msg += " I recommend to overwrite this file."
                raise Exception(msg)
        
        # check the end_ps
        if end_ps != end_time_should:
            msg += (f" The end time of this file {end_ps} is different than the "
                    f"requested {end_time_should=}.")
            if file_exists_policy == 'check_and_overwrite':
                msg += " I will overwrite this file."
                logger.debug(msg)
                return "create"
            else:
                msg += " I recommend to overwrite this file."
                raise Exception(msg)
        
        timesteps = times[1:] - times[:-1]
        timesteps = np.unique(timesteps)
        
        # check the timesteps
        if len(timesteps) == 0:
            msg += (f" However, the file {file} is completely empty. ")
            if file_exists_policy == 'check_and_overwrite':
                msg += " I will overwrite this file."
                logger.debug(msg)
                return "create"
            else:
                msg += " I recommend to overwrite this file."
                raise Exception(msg)
        elif len(timesteps) > 1:
            msg += (f" This file has inconsistent timesteps "
                    f"with varying lengths of {timesteps}. ")
            if file_exists_policy == 'check_and_overwrite':
                msg += " I will overwrite this file."
                logger.debug(msg)
                return "create"
            else:
                msg += " I recommend to overwrite this file."
                raise Exception(msg)
            
        # at this point the timesteps are ok
        timestep = timesteps[0]
        if timestep != dt_should:
            msg += (f" However, the timeteps of the file dt(file)={timestep} is not "
                    f"equal to the requested timestep of dt={dt_should}.")
            if file_exists_policy == 'check_and_overwrite':
                msg += " I will overwrite this file."
                logger.debug(msg)
                return "create"
            else:
                msg += " I recommend to overwrite this file."
                raise Exception(msg)
        else:
            msg += (f" The file is correct.")
            logger.debug(msg)
            return "continue"
        
    elif file_exists_policy == "overwrite":
        return "create"
    
    else:
        raise Exception(f"`file_exists_policy` needs to be one of the following: "
                        f"'raise', 'overwrite', 'continue', 'check_and_continue'. "
                        f"You supplied {file_exists_policy}.")
        
    raise Exception("You should not be able to reach this part of the function.") 
    
    
def get_start_end_in_dt(start_time: float,
                        end_time: float,
                        dt: int
                       ) -> Tuple[int, int]:
    start = int(math.ceil((start_time) / float(dt))) * dt
    end = int(math.floor((end_time) / float(dt))) * dt
    return start, end


def _get_logger(logfile: Path = Path('sim_cleanup.log'),
                loggerName: str = 'SimCleanup',
               ) -> logging.Logger:
    if loggerName in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name=loggerName)
    else:
        logger = logging.getLogger(name=loggerName)
    
        # console
        fmt = "%(name)s %(levelname)8s [%(asctime)s]: %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # file
        fmt = ('%(name)s %(levelname)8s [%(asctime)s] ["%(pathname)s:'
               '%(lineno)s", in %(funcName)s]: %(message)s')
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def check_input_file(input_file: Path,
                     dt: int,
                     max_time: int,
                     previous_file: Optional[Path] = None,
                     previous_start: Optional[int] = None,
                     previous_end: Optional[int] = None,
                     per_file_timestep_policy: PerFileTimestepPolicyType = "raise",
                     inter_file_timestep_policy: InterFileTimestepPolicyType = "raise",
                    ) -> Tuple[int, int, str]:
    logger = logging.getLogger("SimCleanup")
    msg = f"Checking the timesteps of input file: {input_file}."
    if previous_file is not None:
        assert previous_start is not None and previous_end is not None
    logger.debug(f"Comparing to previous file {previous_file} ."
                 f"With start: {previous_start}, and end: {previous_end}.")
    logger.debug(msg)
    start_ps, end_ps, times = get_times_from_file(str(input_file))
    
    # analyze the timesteps of the input file
    timesteps = times[1:] - times[:-1]
    timesteps, counts = np.unique(timesteps, return_counts=True)
    
    # empty sim
    if len(timesteps) == 0:
        if per_file_timestep_policy == "raise":
            msg = (f"The simulation at {input_file} is an 'empty' simulation. Such sims are produced, "
                   f"when gromacs is told to continue from a checkpoint, but the number of steps has "
                   f"reached the number set in the mdp file of the simulation. Gromacs will still "
                   f"write an xtc file with 0 timesteps. Using {per_file_timestep_policy=} caused "
                   f"this to produce an exception. Choose a different `per_file_timestep_policy` "
                   f"to not raise an Exception. 'stop_iter_on_empty' is similar to 'raise' except "
                   f"for this specific error.")
            raise Exception(msg)
        else:
            logger.debug(f"The file {input_file} is empty. Reached end of simulation.")
            return None, None, None, None, "stop_iter", None
        
    # good timesteps
    elif len(timesteps) == 1:
        timesteps = "good"
    
    # bad timesteps do some additional tests
    # maybe also test the median drift velocity
    else:
        debug = ', '.join([f"Timestep of {t} occurs {c} times" for t, c in zip(timesteps, counts)])
        raise NotImplementedError(f"Bad timesteps: {debug}")
        
    if previous_file is not None:
        start_dt = int(math.ceil(start_ps / float(dt))) * dt
        end_dt = int(math.floor(end_ps / float(dt))) * dt
        previous_end_dt = int(math.floor(previous_end / float(dt))) * dt
        
        # calculate differences
        diff = start_dt - previous_end_dt
        end_end_diff = end_dt * dt - previous_end_dt

        # what to do with the difference
        if diff < 0 and end_end_diff < 0:               
            if end_end_diff < 0:
                msg = (f"The file {xtc_file} can be omitted completely. It ends at {end_ps} ps. "
                       f"The previous file {previous_file} ends at {previous_end} ps. So this file contains "
                       f"no additional simulation data. The cause for this might be that {previous_file} "
                       f"did not create a checkpoint (.cpt) file upon completion and gromacs continued "
                       f"at the last available checkpoint.")
                
                if inter_file_timestep_policy == 'raise':
                    msg += (f" Due to the chosen `inter_file_timestep_policy` I have raised an Exception. "
                            f"However, there is a good chance that with `inter_file_timestep_policy=fix_consflicts` "
                            f"you will get the desired result.")
                    raise Exception(msg)
                elif inter_file_timestep_policy == 'ignore':
                    msg += f" Ignoring that discrepancy."
                    logger.debug(msg)
                elif inter_file_timestep_policy == 'fix_conflicts':
                    logger.debug(msg)
                    raise NotImplementedError("There needs to be additional checks here.")
                else:
                    raise Exception(f"`inter_file_timestep_policy` needs to be one of the following: "
                                    f"'raise', 'ignore', 'fix_conflict'. You supplied {inter_file_timestep_policy}.")

                # check the next file and see, if it can be fixed...
        #                     next_start, next_end, _ = get_times_from_file(str(xtc_files[xtc_num + 1]))
        #                     dt_to_next = int(math.ceil(next_start / float(dt))) * dt - previous

        #                     if dt_to_next > 0:
        #                         print(f"The next file {xtc_files[xtc_num + 1]} starts at {next_start}.")


        if diff > dt:
            # check_gmx_times(xtc_files[xtc_num - 1], xtc_file)
            raise Exception(f"The timedelta between {previous_file} and {xtc_file} is not dt={dt}, "
                            f"but {diff} which is larger than the chosen dt "
                            f"(end time of {previous_file}: {previous}, "
                            f"end time of {xtc_file}: {start_dt}). "
                            f"Thus, I can't determine a start time for the trjconv command.")
        if diff < dt:
            start_dt += diff + dt
        if start_dt > max_time:
            timesteps = "stop_iter"
            
        trjconv_b = start_dt
        trjconv_e = end_dt
        
    else:
        trjconv_b, trjconv_e = get_start_end_in_dt(start_ps, end_ps ,dt)
    return start_ps, end_ps, trjconv_b, trjconv_e, timesteps, times


async def get_times_from_file(file: Path, i: int) -> dict[Path, np.ndarray]:
    with XTC(file) as f:
        times = np.fromiter(f, dtype=float)

    xr.Dataset(
        {
            "ps": (["traj", "frame"], [times]),
            "name": (["traj"], [str(file)]),
            "hash": (["traj"], [imohash.hashfile(file, hexdigest=True)])
        },
        coords={
            "frame": np.arange(len(times)),
            "traj": [i]
        }
    )
    return ds


def combine_attrs(variable_attrs, context):
    res = {}
    for d in variable_attrs:
        res.update(d)
    return res


async def write_and_check_times(simulation: tuple[Path, dict[Path, Path]],
                                max_time: int = -1,
                                dt: int = -1,
                                n_atoms: int = -1,
                                pbc: str = "nojump",
                                ) -> bool:
    sim_dir, sim_files = simulation
    data_file = sim_dir / "metadata.nc"
    if data_file.is_file():
        data_file.unlink()
    if data_file.is_file():
        ds = xr.open_dataset(data_file)

        # check existence of all out

        # check hash otherwise calculate new

        # check max_time of all out

        # check df of all out

        # for the remaining files check existence, dt, start, stop,

        # check
        print(ds.frame_no)
        print(ds.attrs)

    else:
        times = await asyncio.gather(
            *[get_times_from_file(s, i) for i, s in enumerate([k for k, v in sim_files.items() if k != "trjcat"])]
        )
        times.extend(await asyncio.gather(
            *[get_times_from_file(s, i + len(times)) for i, s in enumerate([v for k, v in sim_files.items() if k != "trjcat" and v.is_file()])]
        ))
        if sim_files["trjcat"]:
            all_out_file = sim_files["trjcat"]
            if all_out_file.is_file():
                times.append(await get_times_from_file(all_out_file, 1 + len(times)))
        ds = xr.merge(times)
        ds.to_netcdf(data_file, format="NETCDF4", engine="h5netcdf")

    raise NotImplementedError



async def create_ndx_files(simulations: dict[Path, dict[Path, Path]],
                           s: str = "topol.tpr",
                           deffnm: Optional[str] = None,
                           n_atoms: int = -1,
                           ndx_add_group_stdin: str = "",
                           file_exists_policy: FileExistsPolicyType = "raise",
                          ) -> None:
    await asyncio.gather(
        *[create_ndx_file(simulation, s, deffnm, n_atoms, ndx_add_group_stdin, file_exists_policy)
          for simulation in simulations.keys()]
    )


async def create_ndx_file(simulation: Path,
                          s: str = "topol.tpr",
                          deffnm: Optional[str] = None,
                          n_atoms: int = -1,
                          ndx_add_group_stdin: str = "",
                          file_exists_policy: FileExistsPolicyType = "raise",
                         ) -> None:
    if deffnm is not None and s == 'topol.tpr':
        s = deffnm + '.tpr'
    tpr_file = simulation / s
    ndx_file = simulation / "index.ndx"
    logger = _get_logger()
    overwrite = False
    check = False
    if ndx_file.is_file():
        if file_exists_policy == "raise":
            raise Exception(f"File {ndx_file} already exists. "
                            "Due to the chosen `file_exists_policy`={file_exists_policy} "
                            "I have raised an Exception.")
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
            raise Exception(f"Unkown file_exists_policy: {file_exists_policy}")
    else:
        overwrite = True

    if check and n_atoms != -1:
        text = ndx_file.read_text()
        group_name = re.findall(r"\[(.*?)\]", text)[-1].strip()
        text = text.split(']')[-1].strip()
        fields = [row for line in text.splitlines() for row in line.split()]
        if len(fields) != n_atoms:
            if not overwrite:
                raise Exception(f"{ndx_file} indexes the wrong number of atoms. "
                                f"{n_atoms} was requested, but the file contains "
                                f"{len(fields)} atoms for the new group: {group_name}. "
                                f"set `file_exists_policy` to 'overwrite' or "
                                f"'check_and_overwrite' to overwrite this file.")
            else:
                logger.debug(f"{ndx_file} indexes the wrong number of atoms. "
                             f"{n_atoms} was requested, but the file contains "
                             f"{len(fields)} atoms for the new group: {group_name}. "
                             f"I will overwrite this file.")
        else:
            logger.debug(f"{ndx_file} is fine. The group {group_name} has the correct "
                         f"number of atoms: {n_atoms}.")
            return

    if overwrite:
        ndx_file.unlink(missing_ok=True)

    cmd = f"gmx make_ndx -f {tpr_file} -o {ndx_file}"
    ndx_add_group_stdin = ndx_add_group_stdin.encode()
    proc = await asyncio.subprocess.create_subprocess_shell(cmd=cmd,
                                                            stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.PIPE,
                                                            stdin=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate(ndx_add_group_stdin)
    if not ndx_file.is_file():
        print(proc.returncode)
        print(proc.stderr)
        print(proc.stdout)
        print(cmd)
        raise Exception(f"Could not create file {ndx_file}")
    else:
        print(f"Created {ndx_file}")

    text = ndx_file.read_text()
    group_name = re.findall(r"\[(.*?)\]", text)[-1].strip()
    text = text.split(']')[-1].strip()
    fields = [row for line in text.splitlines() for row in line.split()]
    if len(fields) != n_atoms:
        raise Exception(f"{ndx_file} indexes the wrong number of atoms. "
                        f"{n_atoms} was requested, but the file contains "
                        f"{len(fields)} atoms for the new group: {group_name}. "
                        f"You can try and provide a different `ndx_add_group_stdin`."
                        f"Try to find the correct stdin by calling the command: "
                        f"{cmd} manually.")


async def prepare_sim_cleanup(simulations: dict[Path, dict[Path, Path]],
                              max_time: int = -1,
                              dt: int = -1,
                              n_atoms: int = -1,
                              pbc: str = "nojump",
                              ) -> dict:
    plan = await asyncio.gather(
        *[write_and_check_times(simulation, max_time, dt, n_atoms, pbc)
          for simulation in simulations.items()]
    )
    print(plan)
    raise Exception


################################################################################
# Main functions
################################################################################


def cleanup_sims(directories: List[str],
                 out_dir: Union[str, Path],
                 dt: int = -1,
                 max_time: int = -1,
                 n_atoms: int = -1,
                 s: str = 'topol.tpr',
                 x: str = 'traj_comp.xtc',
                 pbc: str = 'nojump',
                 center: str = "center",
                 output_group_and_center: Optional[Union[str, int]] = None,
                 deffnm: Optional[str] = None,
                 trjcat: bool = True,
                 create_pdb: bool = True,
                 create_ndx: bool = False,
                 ndx_add_group_stdin: Optional[str] = None,
                 per_file_timestep_policy: PerFileTimestepPolicyType = "raise",
                 inter_file_timestep_policy: InterFileTimestepPolicyType = "raise",
                 file_exists_policy: FileExistsPolicyType = "raise",
                 clean_copies: bool = False,
                 logfile: Optional[Path] = Path("sim_cleanup.log"),
                 ) -> None:
    """Cleans up your messy simulations.
    
    The `directories` argument can include truncation marks which will keep the
    directory structure up to that point. For example, if you provide
    ['/path/to/./sim_folder/production'] as `directories` and '/home/me/' as
    `out_dir`, the simulation file without solvent can be found in
    '/home/me/sim_folder/production/traj.xtc'.
    
    Args:
        directories (List[str]): A list of strings giving the directories that
            will be searched for simulations containing solvent. Can have a
            truncation mark to define the directory structure in `out_dir`.
        out_dir (Union[str, Path]): The directory to put the solvent-free
            simulations.
        dt (int): The timestep in ps to use for gromacs' `gmx trjconv`.
        max_time (int): The maximum time in ps to use. This time will be
            used as the final -e flag for gromacs' `gmx trjconv`.
        s (str): The name of the topology files in the directories. Similar
            to gromac's -s flag, this defaults to 'topol.tpr'.
        x (str): The name of the compressed trajectory files in the
            directories. Similar to gromacs' -x flag, this defaults to
            'traj_comp.xtc' but 'traj_comp.part0001.xtc', 'traj_comp.part0002.xtc'
            and so on are also included.
        pbc (str): The -pbc flag for gromacs' `gmx trjconv`. Defaults to 'nojump'
            as this is usually the best option for single proteins. Choose 'whole'
            for a simulation with multiple non-connected proteins.
        deffnm (Optional[str]): Similar to gromacs' -deffnm flag. Sets default
            filenames for the .tpr and .xtc files. But is superseded by
            providing non-standard filenames for `s` and `x`.
        trjcat (bool): Whether to concatenate the output files, if the
            simulation is divided into parts (traj_comp.partXXXX.xtc).
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
            
    
    """
    logger = _get_logger()

    # set level
    logger.setLevel(logging.DEBUG)
    logging.StreamHandler.terminator = "\n"
    
    # print a start
    logger.warning("Need to choose what to do with logfile.")
    logger.debug("Started to clean up simulations.")

    # check the input policies
    if per_file_timestep_policy not in PerFileTimestepPolicyType.__args__:
        raise ValueError(f"The `per_file_timestep_policy` needs to be one of the following: "
                         f"{PerFileTimestepPolicyType}, but you provided: {per_file_timestep_policy}")
    if inter_file_timestep_policy not in InterFileTimestepPolicyType.__args__:
        raise ValueError(f"The `inter_file_timestep_policy` needs to be one of the following: "
                         f"{InterFileTimestepPolicyType}, but you provided: {inter_file_timestep_policy}")
    if file_exists_policy not in FileExistsPolicyType.__args__:
        raise ValueError(f"The `file_exists_policy` needs to be one of the following: "
                         f"{FileExistsPolicyType}, but you provided: {file_exists_policy}")
    
    # get the input and the predefined output
    simulations = map_in_and_out_files(directories, out_dir, x, pbc, deffnm, trjcat)
    assert len(simulations) == len(directories)
    logger.debug(f"{len(simulations)} simulations will be cleaned up.")

    # write the ndx files
    if create_ndx:
        asyncio.run(create_ndx_files(simulations, s, deffnm, n_atoms, ndx_add_group_stdin, file_exists_policy))

    # prepeare everything
    # this method filters out what actually needs to be done and whether it is doable
    # out comes a dictionary that can be passed to asyncio
    plan = asyncio.run(prepare_sim_cleanup(simulations, max_time, dt, n_atoms, pbc))
    print(plan)
    raise Exception("STOP")
        
    # iterate over the sims
    for sim_num, (sim_dir, sims) in enumerate(simulations.items()):
        print_sim_name = directories[sim_num]
        if '/./' in print_sim_name:
            print_sim_name = print_sim_name.split('/./')[1]
        else:
            print_sim_name = '.../' + '/'.join(print_sim_name.split('/')[-4:])
        logger.debug(f"{ORDINAL(sim_num + 1)} simulation is {print_sim_name}.")
        
        # find the tpr file in the directory
        if deffnm is not None and s == 'topol.tpr':
            s = deffnm + '.tpr'
        tpr_file = sim_dir / s
        assert tpr_file.is_file(), print(f".tpr file {tpr_file} does not exist.")
        
        # check for duplicated files
        xtc_files = list(sims.keys())
        logger.debug(f"This sim has {len(xtc_files)} .xtc files associated to it.")
        filenames = [f.name for f in xtc_files]
        if len(filenames) != len(set(filenames)):
            duplicate_names = [item for item, count in collections.Counter(filenames).items() if count > 1]
            xtc_files = list(filter(lambda x: x.name not in duplicate_names or 'mannheim' in str(x), xtc_files))
            assert  len(filenames) != len(set(filenames))
            raise Exception(f"Apparently there were duplicate filenames in {sim_dir}. The code to "
                            f"filter duplicates is old and needs a rework.")
            
        # check whether the cat-file exists
        if trjcat:
            all_out_file = sims["trjcat"]
            if (
                (check_all_out_file := check_exisiting_file(
                        all_out_file,
                        dt,
                        max_time,
                        file_exists_policy=file_exists_policy,
                    ))
                    == "continue"
                ):
                continue
                
        # iterate over the sims
        for xtc_num, (xtc_file_in, xtc_file_out) in enumerate(sims.items()):
            logger.debug(f"Clearing solvent form xtc {xtc_num} in {ORDINAL(sim_num + 1)} sim.")
            if xtc_num == 0:
                previous_file, previous_start, previous_end = (None, None, None)
            
            # based on some file-exists policies we can try an early continue
            # however for checking the start and end-time we need to load
            # the xtc_in_file, which would slow us down, if we don't call
            # check_existing_file twice.
            check_out_file = "check"
            if xtc_file_out.is_file():
                if file_exists_policy == "raise":
                    raise Exception(f"File {xtc_file_out} already exists. "
                                    "Due to the chosen `file_exists_policy`={file_exists_policy} "
                                    "I have raised an Exception.")
                elif file_exists_policy == "continue":
                    logger.debug(f"File {xtc_file_out} already exists. Continuing")
                    continue
                elif file_exists_policy == "overwrite" and per_file_timestep_policy == "ignore":
                    check_out_file = "create"
                    pass
            else:
                logger.debug(f"I will create the file {xtc_file_out}")
                
            # we check for that we need the xtc_file_in
            start_ps, end_ps, trjconv_b, trjconv_e, result, times = check_input_file(
                xtc_file_in,
                dt,
                max_time,
                previous_file,
                previous_start,
                previous_end,
                per_file_timestep_policy,
                inter_file_timestep_policy,
            )
            
            # if the file is empty, we reached the end of the simulation
            if result == "stop_iter":
                if end_ps < max_time:
                    logger.error(f"The last file {xtc_file_in} in the sim folder {sim_dir} did not reach the "
                                 f"requested simulation length of {max_time} ps. It seems the "
                                 f"simulation is not yet finished. This doesn't leed to termination of the "
                                 f"remaining cleanup, but you want to have a look at this specific simulation.")
                break

            if check_out_file != "create":
                if (
                    (check_out_file := check_exisiting_file(
                            xtc_file_out,
                            dt,
                            end_time_should=trjconv_e,
                            start_time_should=trjconv_b,
                            file_exists_policy=file_exists_policy,
                            start_ps=start_ps,
                            end_ps=end_ps,
                            times=times,
                        ))
                        == "continue"
                    ):
                    continue
                
            # create the out-directory if it does not exist
            if not xtc_file_out.parent.is_dir():
                logger.debug(f"Creating the output directory: {xtc_file_out.parent}")
                xtc_file_out.parent.mkdir(parents=True, exist_ok=True)
                
            # overwrite the previous stuff
            if xtc_num > 0:
                previous_file = deepcopy(xtc_file_in)
                previous_start = deepcopy(start_ps)
                previous_end = deepcopy(end_ps)

            cmd = f'gmx trjconv -f {xtc_file_in} -s {tpr_file} -o {xtc_file_out} -dt {dt} -pbc {pbc} -b {trjconv_b} -e {trjconv_e}'
            if check_out_file == "create":
                proc = run(cmd, stdout=PIPE,
                           stderr=PIPE,
                           input='1\n',
                           universal_newlines=True,
                           shell=True,
                          )
                if not xtc_file_out.is_file():
                    print(proc.returncode)
                    print(proc.stderr)
                    print(proc.stdout)
                    print(cmd)
                    raise Exception(f"Could not create file {xtc_file_out}")
                else:
                    logger.debug(f"Created {xtc_file_out}")
                    
        # make the trjcat here. We should have passed all tests. So all should be good
        if trjcat:
            out_files = ' '.join(list(map(str, sims.values())))
            cmd = f"gmx trjcat -f {out_files} -o {all_out_file}"
            proc = run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
            if not all_out_file.is_file():
                print(proc.returncode)
                print(proc.stderr)
                print(proc.stdout)
                print(cmd)
                raise Exception(f"Could not create file {all_out_file}")
            else:
                logger.debug(f"Created {all_out_file}")
                
        # as a last part create a pdb in the directory
        if create_pdb:
            pdb_file = list(sims.values())[0].parent / f"start.pdb"
            cmd = f'gmx trjconv -f {all_out_file} -s {tpr_file} -o {pdb_file} -dump 0 -pbc {pbc}'
            proc = run(cmd, stdout=PIPE,
                       stderr=PIPE,
                       input='1\n',
                       universal_newlines=True,
                       shell=True,
                      )
            if not pdb_file.is_file():
                print(proc.returncode)
                print(proc.stderr)
                print(proc.stdout)
                print(cmd)
                raise Exception(f"Could not create file {pdb_file}")
            else:
                logger.debug(f"Created {pdb_file}")
                
        if clean_copies:
            copy_files = list(list(sims.values())[0].parent.glob('#*'))
            logger.debug(f"Deleting {len(copy_files)} copy files (filenames like this: {copy_files[0]})")
            for f in copy_files:
                f.unlink()
            
    logger.debug("All finished. Rejoice.")
    

################################################################################
# Argparse and make it a script
################################################################################


# %% for pycharm scientifc mode
if __name__ == "__main__":
    if 'update_gmx_environ' not in globals():
        from cleanup_sims import update_gmx_environ, cleanup_sims
        from pathlib import Path

    update_gmx_environ('2022.2')

    # collect sim dirs
    simulation_dirs = list(Path('/mnt/scc3/kevin/archive').glob('tetraUb/*tetraUBQ*/'))

    # add truncation marks
    simulation_dirs = ['/' + '/'.join([*d.parts[1:-1], '.', d.parts[-1]]) for d in simulation_dirs[:3]]

    # run
    cleanup_sims(directories=simulation_dirs,
                 out_dir='/home/kevin/projects/molsim/tetraUb',
                 dt=100,
                 max_time=50000000,
                 n_atoms=652,
                 output_group_and_center="Protein_GLQ_LYQ",
                 create_ndx=True,
                 ndx_add_group_stdin='1 | 13 | 14\nq\n',
                 file_exists_policy="check_and_overwrite",
                )
