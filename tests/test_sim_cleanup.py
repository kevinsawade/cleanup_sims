"""Unittests for sim_cleanup.

## A good simulation
In the following test cases, I will discern between timestamp and timestep.
A timestamp is the time of the frame (xyz coordinates of the atoms). A timestep
is the time between two timestamps. Based on this a normal simulation is
made up from frames like this:

################################################################################
Simulation 1:
timestamps: |0   10   20   30   40   50   60   70   80   90   100   110   120
timeline:   |<------------------------file1.xtc---------------------------->|
timesteps:  |  10   10   10   10   10   10   10   10   10   10    10   10

max_time_sim: 120
unique_dt_sim: 10
################################################################################


### Overview of operations

The following operations can be done on such simulations:
    * Remove solvent (this is always possible, as long as solvent and protein are
        indexed).
    * Choose a different max_time < max_tim_sim.
    * Choose a different dt with unique_dt_sim % dt == 0.

## Files

However, not all simulations are made up of a single file with nice timesteps.
They might consist of multiple files with a different number of errors. Let's first
talk about the files.

Normally, GROMACS simulations run in -append mode, this means
that, if the mdrun process gets interrupted and then another process is started, the
program continues to write into the same xtc/trr/edr/log files. However, you might
also run mdrun with the -noappend option. The first files produced by the prorgram
are: (traj.part0001.trr, traj_comp.part0001.xtc, ener.part0001.edr). The next files
are then (traj.part0002.xtc etc.). However, there's a mixed option. Let's call it
append-noappend. This happens, if the first call to mdrun was done with -append, and
consecutive calls to mdrun are done with -noappend. This will result in files with
this pattern: traj_comp.xtc, traj_comp.part0002.xtc, traj_comp.part0003.xtc. In the
remainder of this explanation, I will name the files only file1.xtc, file2.xtc, etc.
The distinction between append/noappend/append-noappend is easy to implement.

## Simulations with multiple files.

Some simulations might have multiple files:

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

These simulations are also pretty easy to treat. With the trjconv and trjcat
command, they can be sliced as needed. Let's say the user wants mas_time = 100 ps
and dt = 20 ps. The file1.xtc will be used as input for trconv and a new file with
the timestamps (0, 20, 40) will be written. From file2.xtc the timestamps (60, 80, 100)
will be written. File3.xtc can be ignored. After the trjcat simulation 2 looks like this:

################################################################################
Simulation 2 after trjconv:
timestamps: |0        20        40
timeline:   |<-file1_nosolv.xtc->|
timesteps:  |    20        20

timestamps:                              |60        80        100
timeline:                                |<--file2_nosolv.xtc-->|
timesteps:                               |     20        20
################################################################################

A call to gmx trjcat then will concatenate these files and result in:

################################################################################
Simulation 2 after trjcat:
timestamps: |0        20        40        60        80        100
timeline:   |<---------------file_nosolv.xtc------------------->|
timesteps:  |    20        20        20        20        20
################################################################################

After this the cleanup is finished. Problems come from simulations which can look
like this:

### Problem case 1: Overlapping timestamps without discarding

To be honest, I have no idea, how these arrangements of files can be written
by gmx mdrun. I am only here to clean them up.

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

Let's say, we want max_time = 120 ps and dt = 20 ps. In that case, we need to
tell gmx trjconv to use a different start time (flag -b) for file2.xtc. The commands
are:
    gmx trjconv -f file1.xtc -dt 20 -e 40
    gmx trjconv -f file2.xtc -dt 20 -b 60 -e 100
    gmx trjconv -f file3.xtc -dt 20 -b 120 -e 120

### Problem case 2: Uneven timesteps with overlapping timestamps

This is another problem, which I don't have the willpower to trace back. Some
gromacs simulations I did, had timesteps tike these:

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

These input files can be used without problems.

### Problem case 4: Uneven timesteps with breaking alterations

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

These input files can not be used to yield a concatenated trajectory with dt = 20 ps.
The timestamps 60 and 80 are missing. There are simply no atomic coordinates
at the respective timestamps available in the files.

### Problem case 5: Missing times between files.

Another breaking problem is, when timestamps between trajectories are missing.

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

It goes without saying, that these input files can not be used.

### Problem case 6: Redundant files.

There is also the possibility, that the input files contain redundant files.

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

In such cases, the file2.xtc can safely be discarded.

### Problem case 7 and 8: One-timestamp-zero-timestap trajs

Some xtc files written by gromacs contain only a single timestamp. Thus, they can
only be used in edge cases.

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

With dt = 20 ps, the file2.xtc can be discarded, with dt = 10 ps, it can not be
discarded. In this case he commands for these files are:
    gmx trjconv -f file1.xtc -dt 10 -e 80
    gmx trjconv -f file2.xtc -dt 10 -b 90 -e 90
    gmx trjconv -f file3.xtc -dt 10 -b 100 -e 120

"""
import shutil
import pytest
import coverage
from pathlib import Path
import datetime
import hypothesis
from time import sleep
import mdtraj as md
from cleanup_sims.cleanup_sims import cleanup_sims
from cleanup_sims.cleanup_sims import update_gmx_environ
from requests import get
ip = get('https://api.ipify.org').content.decode('utf8')
if ip.startswith("134.34"):
    update_gmx_environ("2022.3")


class TestSimCleanup:
    input_dir = Path(__file__).resolve().parent / "data/input_sims"
    input_tpr_file = Path(__file__).resolve().parent / "data/asp5.tpr"
    input_xtc_file = Path(__file__).resolve().parent / "data/asp5.xtc"
    output_dir = Path(__file__).resolve().parent / "data/output_sims"
    log_file = Path(__file__).resolve().parent / "sim_cleanup.log"

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.input_dir)
        shutil.rmtree(cls.output_dir)
                shutil.rmtree(path_object)

    def setup_method(self) -> None:
        # create the input directory
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def test_simulation_all_good(self):
        # name the file traj_comp.xtc and topol.tpr
        input_dir = self.input_dir / "nested/directory/structure"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.input_tpr_file, input_dir / "topol.tpr")
        shutil.copyfile(self.input_xtc_file, input_dir / "traj_comp.xtc")
        input_dir_ = str(self.input_dir) + "/nested/./directory/structure"
        cleanup_sims(directories=[input_dir_],
                     out_dir=self.output_dir,
                     dt=20,
                     max_time=140,
                     center=True,
                     create_pdb=True,
                     file_exists_policy="overwrite",
                     )

        out_file = self.output_dir / "directory/structure/traj_nojump.xtc"
        pdb_file = self.output_dir / "directory/structure/start.pdb"
        assert self.output_dir.is_dir()
        assert out_file.is_file()
        assert pdb_file.is_file()
        test_file = md.load(str(out_file), top=str(pdb_file))
        assert len(test_file) == 8, print(out_file, pdb_file, test_file)


    def test_assert_true(self):
        assert False

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


