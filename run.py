#!/usr/bin/env python
"""
run.py: main script to run tasks in this project
"""

import scripttool
import sys

scripttool.set_output_dir("results")
options = {}
scripttool.set_options(options)

# import and register various experiments
import tasks

if __name__ == "__main__":
    scripttool.main()


