# Tvastar
#
# Copyright (c) 2026 Aravind Sankaran
# Developed in Aachen, Germany.
#
# Licensed under the BSD 3-Clause License.
# See LICENSE file in the project root for full license information.

import os

# get abs path of this file
this_dir = os.path.dirname(os.path.abspath(__file__))
tvst_eg_data = os.getenv('TVST_EG_DATA', os.path.join(this_dir, 'eg_data'))