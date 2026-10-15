# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cst_python_api 

# Obtain path of the folder containing the example CST files
dataFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CST_Files'))