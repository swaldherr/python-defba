How to use this code
====================

Preliminaries
-------------

Download the pybrn package from 

    http://pybrn.sourceforge.net

Either install the package into your local python installation, or simply copy
the ``brn`` directory into this directory.

You will also need the cvxopt package from

    http://cvxopt.org

For some of the computations, the glpk package from

    http://www.gnu.org/software/glpk/

may be required.

Execution
---------

In order to run all optimizations, execute the command

    python run.py --task=all --log --export=png

Output files will be generated in a subdirectory called ``results``.

You can also select a specific task to be run by executing the command

    python run.py --task=<task-id> --log --export=png

where <task-id> should be replaced by the identifier of the task.  You can get a
list of all implemented tasks by executing

    python run.py --print-tasks

Diving into the code
--------------------

A good point to start exploring the code is in the definition of the individual
tasks. These are contained in the files ``tasks/*.py``.  Tasks are defined as
subclasses of ``scripttool.Task``.  For each task, there is a ``run()`` method
which contains the code that is executed for the task.


