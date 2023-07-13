"""# In-depth explanation

## Basic concepts

### A good simulation
In the following test cases, I will discern between timestamp and timestep.
A timestamp is the time of the frame (xyz coordinates of the atoms). A timestep
is the time between two timestamps. Based on this a normal simulation is
made up from frames like this:

```
################################################################################
Simulation 1:
timestamps: |0   10   20   30   40   50   60   70   80   90   100   110   120
timeline:   |<------------------------file1.xtc---------------------------->|
timesteps:  |  10   10   10   10   10   10   10   10   10   10    10   10

max_time_sim: 120
unique_dt_sim: 10
################################################################################
```

### Overview of operations

The following operations can be done on such simulations:
    * Remove solvent (this is always possible, as long as solvent and protein are
        indexed).
    * Choose a different max_time < max_tim_sim.
    * Choose a different dt with unique_dt_sim % dt == 0.

### Files

However, not all simulations are made up of a single file with nice timesteps.
They might consist of multiple files with a different number of errors. Let's first
talk about the files.

Normally, GROMACS simulations run in -append mode, this means
that, if the mdrun process gets interrupted and then another process is started, the
program continues to write into the same xtc/trr/edr/log files. However, you might
also run mdrun with the -noappend option. The first files produced by the prorgram
are: (traj.part0001.trr, traj_comp.part0001.xtc, ener.part0001.edr). The next files
are then (traj.part0002.xtc, etc.). However, there's a mixed option. Let's call it
append-noappend. This happens, if the first call to mdrun was done with -append, and
consecutive calls to mdrun are done with -noappend. This will result in files with
this pattern: traj_comp.xtc, traj_comp.part0002.xtc, traj_comp.part0003.xtc. In the
remainder of this explanation, I will name the files only file1.xtc, file2.xtc, etc.
The distinction between append/noappend/append-noappend is easy to implement.

## Cases

Some simulations might have multiple files:

```
################################################################################
Simulation 2:
timestamps: |0   10   20   30   40   50
timeline:   |<------file1.xtc-------->|
timesteps:  |  10   10   10   10   10

timestamps:                              |60   70   80   90   100
timeline:                                |<------file2.xtc----->|
timesteps:                               |   10   10   10   10

timestamps:                                                      |110        120
timeline:                                                        |<-file3.xtc->|
timesteps:                                                       |     10

max_time_sim: 120
unique_dt_sim: 10
################################################################################
```

These simulations are also pretty easy to treat. With the trjconv and trjcat
command, they can be sliced as needed. Let's say the user wants max_time = 100 ps
and dt = 20 ps. The file1.xtc will be used as input for trjconv and a new file with
the timestamps (0, 20, 40) will be written. From file2.xtc the timestamps (60, 80, 100)
will be written. File3.xtc can be ignored. Before the trjcat simulation 2 looks like this:

```
################################################################################
Simulation 2 after trjconv:
timestamps: |0        20        40
timeline:   |<-file1_nosolv.xtc->|
timesteps:  |    20        20

timestamps:                              |60        80        100
timeline:                                |<--file2_nosolv.xtc-->|
timesteps:                               |     20        20
################################################################################
```

A call to gmx trjcat then will concatenate these files and result in:

```
################################################################################
Simulation 2 after trjcat:
timestamps: |0        20        40        60        80        100
timeline:   |<---------------file_nosolv.xtc------------------->|
timesteps:  |    20        20        20        20        20
################################################################################
```

After this the cleanup is finished. Problems come from simulations which can look
like this:

### Problem case 1: Overlapping timestamps without discarding

To be honest, I have no idea, how these arrangements of files can be written
by gmx mdrun. I am only here to clean them up.

```
################################################################################
Simulation 3:
timestamps: |0   10   20   30   40   50
timeline:   |<------file1.xtc-------->|
timesteps:  |  10   10   10   10   10

timestamps:                30   40   50   60   70   80   90   100
timeline:                  |<--------------file2.xtc----------->|
timesteps:                 |   10   10   10   10  10   10   10

timestamps:                                                      |110        120
timeline:                                                        |<-file3.xtc->|
timesteps:                                                       |     10

max_time_sim: 120
unique_dt_sim: 10
################################################################################
```

Let's say, we want max_time = 120 ps and dt = 20 ps. In that case, we need to
tell gmx trjconv to use a different start time (flag -b) for file2.xtc. The commands
are:
    gmx trjconv -f file1.xtc -dt 20 -e 40
    gmx trjconv -f file2.xtc -dt 20 -b 60 -e 100
    gmx trjconv -f file3.xtc -dt 20 -b 120 -e 120

### Problem case 2: Uneven timesteps with overlapping timestamps

This is another problem, which I don't have the willpower to trace back. Some
gromacs simulations I did, had timesteps tike these:

```
################################################################################
Simulation 3:
timestamps: |0   8    20   28   40   48
timeline:   |<------file1.xtc-------->|
timesteps:  |  8   12    8    12   8

timestamps:                28   40   48   60   68   80   88   100
timeline:                  |<--------------file2.xtc----------->|
timesteps:                 |  12   8    12   8   12    8   12

timestamps:                                                      |108        120
timeline:                                                        |<-file3.xtc->|
timesteps:                                                       |     12

max_time_sim: 120
unique_dt_sim: [8, 12]
################################################################################
```

For this simulation **Choosing a dt of 10 ps is not possible**. However, choosing a
dt of 20 ps is still possible, because of how the timestamps are laid out. Cleanup_sims
checks for that condition by asserting the following expression:

.. code-block::
    :language: python:

    assert len(np.arange(0, max_time, dt)) == np.unique(np.hstack([file.times for file in files])) % dt == 0).sum()

Doing so, we can test before calling any command, whether the requested dt can be
realized with the provided data.

### Problem case 3: Uneven timesteps with overlapping trajectories

The uneven timesteps can also propagate differently:

```
################################################################################
Simulation 4:
timestamps: |0   8    20   28   40   48
timeline:   |<------file1.xtc-------->|
timesteps:  |  8   12    8    12   8

timestamps:                     40   52   60   72   80   92   100
timeline:                       |<---------file2.xtc----------->|
timesteps:                      |  12    8   12   8   12    8

timestamps:                                                      |100  108   120
timeline:                                                        |<-file3.xtc->|
timesteps:                                                       |   8    12

max_time_sim: 120
unique_dt_sim: [8, 12]
################################################################################
```

These input files can be used without problems.

### Problem case 4: Uneven timesteps with breaking alterations

```
################################################################################
Simulation 5:
timestamps: |0   8    20   28   40   48
timeline:   |<------file1.xtc-------->|
timesteps:  |  8   12    8    12   8

timestamps:                     40   52   64   78   86   92   100
timeline:                       |<---------file2.xtc----------->|
timesteps:                      |  12   12   12   8    8    8

max_time_sim: 120
unique_dt_sim: [8, 12]
################################################################################
```

These input files can **not be used to yield a concatenated trajectory** with dt = 20 ps.
The timestamps 60 and 80 are missing. There are simply no atomic coordinates
at the respective timestamps available in the files.

### Problem case 5: Missing times between files.

Another breaking problem is, when timestamps between trajectories are missing.

```
################################################################################
Simulation 6:
timestamps: |0   10   20   30   40   50
timeline:   |<------file1.xtc-------->|
timesteps:  |  10   10   10   10   10

timestamps:                                       | 90      100
timeline:                                         |<-file2.xtc->|
timesteps:                                        |     10

max_time_sim: 120
unique_dt_sim: 10
################################################################################
```

It goes without saying, that these input files can not be used.

### Problem case 6: Redundant files.

There is also the possibility, that the input files contain redundant files.

```
################################################################################
Simulation 7:
timestamps: |0   10   20   30   40   50   60   70   80   90
timeline:   |<----------------file1.xtc------------------>|
timesteps:  |  10   10   10   10   10   10   10   10   10

timestamps:                30   40   50   60   70
timeline:                  |<-----file2.xtc---->|
timesteps:                 |  10   10   10   10

timestamps:                                                    |100    110   120
timeline:                                                      |<--file3.xtc-->|
timesteps:                                                     |   10     10

max_time_sim: 120
unique_dt_sim: 10
################################################################################
```

In such cases, the file2.xtc can safely be discarded.

### Problem case 7 and 8: One-timestamp-zero-timestap trajs

Some xtc files written by gromacs contain only a single timestamp. Thus, they can
only be used in edge cases.

```
################################################################################
Simulation 8:
timestamps: |0   10   20   30   40   50   60   70   80
timeline:   |<--------------file1.xtc--------------->|
timesteps:  |  10   10   10   10   10   10   10   10

timestamps:                                              90
timeline:                                          |<file2.xtc>|
timesteps:

timestamps:                                                    |100    110   120
timeline:                                                      |<--file3.xtc-->|
timesteps:                                                     |   10     10

max_time_sim: 120
unique_dt_sim: 10
################################################################################
```

With dt = 20 ps, the file2.xtc can be discarded, with dt = 10 ps, it can not be
discarded. In this case the commands for these files are:
    gmx trjconv -f file1.xtc -dt 10 -e 80
    gmx trjconv -f file2.xtc -dt 10 -b 90 -e 90
    gmx trjconv -f file3.xtc -dt 10 -b 100 -e 120

## Can we trust gromacs times and coordinates

"""

import shutil
from pathlib import Path
import numpy as np
import coverage
import re
import shlex
from hypothesis import given
from hypothesis.strategies import integers
import mdtraj as md
import pytest
from cleanup_sims.cleanup_sims import cleanup_sims, update_gmx_environ
from requests import get


ip = get("https://api.ipify.org").content.decode("utf8")
if ip.startswith("134.34"):
    update_gmx_environ("2022.3")


class TestSimCleanup:
    input_dir = Path(__file__).resolve().parent / "data/input_sims"
    input_tpr_file = Path(__file__).resolve().parent / "data/asp5.tpr"
    input_xtc_file = Path(__file__).resolve().parent / "data/asp5.xtc"
    input_gro_file = Path(__file__).resolve().parent / "data/asp5.gro"
    output_dir = Path(__file__).resolve().parent / "data/output_sims"
    log_file = Path(__file__).resolve().parent / "sim_cleanup.log"

    @classmethod
    def setup_class(cls):
        if cls.input_dir.is_dir():
            shutil.rmtree(cls.input_dir)
        if cls.output_dir.is_dir():
            shutil.rmtree(cls.output_dir)

    def teardown_method(self):
        if self.input_dir.is_dir():
            shutil.rmtree(self.input_dir)
        if self.output_dir.is_dir():
            shutil.rmtree(self.output_dir)

    def setup_method(self) -> None:
        # create the input directory
        self.input_dir.mkdir(parents=True, exist_ok=True)

    @given(integers())
    def test_ordinal_numbers(self, i):
        from cleanup_sims.cleanup_sims import ORDINAL
        test = ORDINAL(i)
        test = int(re.findall(r'-?\d+', test)[0])
        assert test == i

    def test_no_dt_only_pbc(self):
        assert False

    def test_parser(self):
        from cleanup_sims.cleanup_sims import main

        # name the file traj_comp.xtc and topol.tpr
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)

        # create case 1 simulation
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")
        shutil.copyfile(self.input_xtc_file, input_dir / "traj_comp.xtc")

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cmd = f"-d {input_dir_} -o {self.output_dir} -dt 20 -max 140 -center -create-pdb"
        main(shlex.split(cmd))

        # test
        out_file = self.output_dir / "directory/structure/traj_clean.xtc"
        pdb_file = self.output_dir / "directory/structure/start.pdb"
        assert self.output_dir.is_dir()
        assert out_file.is_file()
        assert pdb_file.is_file()
        test_file = md.load(str(out_file), top=str(pdb_file))
        assert len(test_file) == 8, print(out_file, pdb_file, test_file)

    def test_simulation_all_good(self):
        # name the file traj_comp.xtc and topol.tpr
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)

        # create case 1 simulation
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")
        shutil.copyfile(self.input_xtc_file, input_dir / "traj_comp.xtc")

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cleanup_sims(
            directories=[input_dir_],
            out_dir=self.output_dir,
            dt=20,
            max_time=140,
            center=True,
            create_pdb=True,
        )

        # test
        out_file = self.output_dir / "directory/structure/traj_clean.xtc"
        pdb_file = self.output_dir / "directory/structure/start.pdb"
        assert self.output_dir.is_dir()
        assert out_file.is_file()
        assert pdb_file.is_file()
        test_file = md.load(str(out_file), top=str(pdb_file))
        assert len(test_file) == 8, print(out_file, pdb_file, test_file)

    def test_discontinuity_between_trajs(self):
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")

        # create case 2 simulation
        full_traj = md.load(str(self.input_xtc_file), top=str(self.input_gro_file))
        assert len(full_traj[:6]) == 6
        assert full_traj[:6].time[-1] == 50
        full_traj[:6].save_xtc(str(input_dir / "traj_comp.part0001.xtc"))
        full_traj[11:13].save_xtc(str(input_dir / "traj_comp.part0002.xtc"))

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        with pytest.raises(Exception):
            cleanup_sims(
                directories=[input_dir_],
                out_dir=self.output_dir,
                dt=20,
                center=True,
                create_pdb=True,
            )

    def test_simulation_made_from_multiple_files_no_trjcat_with_drift(self):
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")

        # create case 2 simulation
        full_traj = md.load(str(self.input_xtc_file), top=str(self.input_gro_file))
        assert len(full_traj[:6]) == 6
        assert full_traj[:6].time[-1] == 50
        raise Exception("Make this have a atomic drift")
        full_traj[:6].save_xtc(str(input_dir / "traj_comp.part0001.xtc"))
        full_traj[6:11].save_xtc(str(input_dir / "traj_comp.part0002.xtc"))
        full_traj[11:13].save_xtc(str(input_dir / "traj_comp.part0003.xtc"))

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cleanup_sims(
            directories=[input_dir_],
            out_dir=self.output_dir,
            dt=20,
            center=True,
            create_pdb=True,
            trjcat=False,
        )

    def test_simulation_made_from_multiple_files_with_trjcat_with_drift(self):
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")

        # create case 2 simulation
        full_traj = md.load(str(self.input_xtc_file), top=str(self.input_gro_file))
        assert len(full_traj[:6]) == 6
        assert full_traj[:6].time[-1] == 50
        raise Exception("Make this have a atomic drift")
        full_traj[:6].save_xtc(str(input_dir / "traj_comp.part0001.xtc"))
        full_traj[6:11].save_xtc(str(input_dir / "traj_comp.part0002.xtc"))
        full_traj[11:13].save_xtc(str(input_dir / "traj_comp.part0003.xtc"))

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cleanup_sims(
            directories=[input_dir_],
            out_dir=self.output_dir,
            dt=20,
            center=True,
            create_pdb=True,
            trjcat=True,
        )

    def test_simulation_made_from_multiple_files_no_trjcat(self):
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")

        # create case 2 simulation
        full_traj = md.load(str(self.input_xtc_file), top=str(self.input_gro_file))
        assert len(full_traj[:6]) == 6
        assert full_traj[:6].time[-1] == 50
        full_traj[:6].save_xtc(str(input_dir / "traj_comp.part0001.xtc"))
        full_traj[6:11].save_xtc(str(input_dir / "traj_comp.part0002.xtc"))
        full_traj[11:13].save_xtc(str(input_dir / "traj_comp.part0003.xtc"))

        # test that traj_comp.part0003.xtc is correct
        test = md.load(str(input_dir / "traj_comp.part0003.xtc"), top=str(self.input_gro_file))

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cleanup_sims(
            directories=[input_dir_],
            out_dir=self.output_dir,
            dt=20,
            center=True,
            create_pdb=True,
            trjcat=False,
        )

        # test
        out_file = self.output_dir / "directory/structure/traj_comp_clean.part0001.xtc"
        pdb_file = self.output_dir / "directory/structure/start.pdb"
        assert self.output_dir.is_dir()
        assert out_file.is_file()
        assert pdb_file.is_file()

        traj1 = md.load(str(out_file), top=str(pdb_file))
        assert traj1.time[1] - traj1.time[0] == 20
        assert len(traj1) == 3

        out_file = self.output_dir / "directory/structure/traj_comp_clean.part0002.xtc"
        assert out_file.is_file()
        traj1 = md.load(str(out_file), top=str(pdb_file))
        assert traj1.time[1] - traj1.time[0] == 20
        assert len(traj1) == 3

        out_file = self.output_dir / "directory/structure/traj_comp_clean.part0003.xtc"
        assert out_file.is_file()
        traj1 = md.load(str(out_file), top=str(pdb_file))
        assert len(traj1.time) == 1
        assert len(traj1) == 1

    def test_simulation_made_from_multiple_files_with_trjcat(self):
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")

        # create case 2 simulation
        full_traj = md.load(str(self.input_xtc_file), top=str(self.input_gro_file))
        assert len(full_traj[:6]) == 6
        assert full_traj[:6].time[-1] == 50
        full_traj[:6].save_xtc(str(input_dir / "traj_comp.part0001.xtc"))
        full_traj[6:11].save_xtc(str(input_dir / "traj_comp.part0002.xtc"))
        full_traj[11:13].save_xtc(str(input_dir / "traj_comp.part0003.xtc"))

        # run
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cleanup_sims(
            directories=[input_dir_],
            out_dir=self.output_dir,
            dt=20,
            center=True,
            create_pdb=True,
            trjcat=True,
        )

        # test
        out_file = self.output_dir / "directory/structure/traj_comp_clean.part0001.xtc"
        pdb_file = self.output_dir / "directory/structure/start.pdb"
        assert self.output_dir.is_dir()
        assert not out_file.is_file()
        assert pdb_file.is_file()
        out_file = self.output_dir / "directory/structure/traj_clean.xtc"
        assert out_file.is_file()

        traj = md.load(str(out_file), top=str(pdb_file))
        assert traj.time[1] - traj.time[0] == 20
        assert len(traj) == 7

    def test_no_output_dir_with_trjcat(self):
        input_dir = self.input_dir / "inp"

    def test_no_output_dir_no_trjcat(self):
        pass

    def test_identify_wrong_times_for_correct_coordinates(self):
        input_dir1 = self.input_dir / "nested/directory/structure"
        input_dir2 = self.input_dir / "also_a_sim"
        input_dir1.mkdir(parents=True, exist_ok=True)
        input_dir2.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir1 / "topol.tpr")
        shutil.copyfile(self.input_tpr_file, input_dir2 / "topol.tpr")

        # define the traj
        full_traj = md.load(str(self.input_xtc_file), top=str(self.input_gro_file))
        assert len(full_traj[:6]) == 6
        assert full_traj[:6].time[-1] == 50

        # times are correct, coordinates are false
        traj1 = full_traj[[0, 2, 3, 5, 6, 8, 9]]
        traj1.time = np.array([0, 10, 20, 30, 40, 50, 60])
        traj2 = full_traj[[11, 12, 14, 15, 17, 18]]
        traj2.time = np.array([70, 80, 90, 100, 110, 120])
        traj3 = full_traj[[20, 21, 23]]
        traj3.time = np.array([130, 140, 150])
        traj1.save_xtc(str(input_dir1 / "traj_comp.part0001.xtc"))
        traj2.save_xtc(str(input_dir1 / "traj_comp.part0002.xtc"))
        traj3.save_xtc(str(input_dir1 / "traj_comp.part0003.xtc"))

        # times are false, coordinates are correct
        traj1 = full_traj[:6]
        traj1.time = np.array([0, 8, 20, 28, 40, 48])
        traj1.save_xtc(str(input_dir2 / "traj_comp.part0001.xtc"))
        traj2 = full_traj[6:11]
        traj2.time = np.array([60, 68, 80, 88, 100])
        traj2.save_xtc(str(input_dir2 / "traj_comp.part0002.xtc"))
        traj3 = full_traj[11:13]
        traj3.time = np.array([108, 120])
        traj3.save_xtc(str(input_dir2 / "traj_comp.part0003.xtc"))

        test = md.load(str(input_dir2 / "traj_comp.part0001.xtc"), top=str(self.input_gro_file))
        assert np.array_equal(test.time, np.array([0, 8, 20, 28, 40, 48]))

        # run
        input_dir1_ = str(self.input_dir) + "/nested/./directory/structure"
        input_dir2_ = str(self.input_dir) + "/also_a_sim"
        cleanup_sims(
            directories=[input_dir1_],
            out_dir=self.output_dir,
            dt=10,
            center=True,
            create_pdb=True,
            trjcat=True,
            loglevel=0
        )

        # traj1 = full_traj[:6]
        # traj1.time = np.array([0, 2, 10, 12, 20, 22])
        # traj1.save_xtc(str(input_dir / "traj_comp.part0001.xtc"))
        # traj2 = full_traj[6:11]
        # traj2.time = np.array([30, 32, 40, 42, 50])
        # traj2.save_xtc(str(input_dir / "traj_comp.part0002.xtc"))
        # traj3 = full_traj[11:13]
        # traj3.time = np.array([52, 60])
        # traj3.save_xtc(str(input_dir / "traj_comp.part0003.xtc"))


    def test_get_lsb(self):
        from cleanup_sims.cleanup_sims import get_lsb

        out = get_lsb()
        assert isinstance(out, str)


if __name__ == "__main__":
    # start coverage
    this_dir = Path(__file__).resolve().parent
    config_file = str(Path(__file__).resolve().parent.parent / "pyproject.toml")
    cov = coverage.Coverage(config_file=config_file)
    cov.start()

    # run pytest
    pytest.main()

    # # find the tests
    # loader = unittest.TestLoader()
    # suite = loader.discover(
    #     start_dir=str(this_dir),
    #     top_level_dir=str(this_dir.parent),
    # )
    #
    # # create an html test runner
    # now = datetime.now().astimezone().replace(microsecond=0).isoformat()
    # runner = HtmlTestRunner.HTMLTestRunner(
    #     output=str(this_dir.parent / "docs/build/static"),
    #     report_title=f"EncoderMap Unittest Report from {now}",
    #     report_name="html_test_runner_report",
    #     combine_reports=True,
    #     add_timestamp=False,
    #     buffer=True,
    # )

    # run the tests
    # result = runner.run(suite)

    # stop coverage
    cov.stop()
