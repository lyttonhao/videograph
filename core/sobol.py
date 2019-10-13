# !/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

import os
import tempfile
import numpy as np

def sobol_generate(n_dim, n_point, n_skip=0):
    if n_dim > 1111:
        raise Exception('This program supports sobol sequence of dimension up to 1111')
    while True:
        try:
            sequence_file = tempfile.NamedTemporaryFile('r')
            filename = sequence_file.name
            cmd = os.path.join(os.path.split(os.path.abspath(os.path.realpath(__file__)))[0], 'sobol_c/sobol')
            cmd += ' ' + str(int(n_dim)) + ' ' + str(int(n_point)) + ' ' + str(int(n_skip)) + ' ' + filename
            os.system(cmd)
            sequence = np.fromfile(sequence_file.file, dtype=np.float64).astype(np.float32)
            sequence = sequence.reshape([n_point, n_dim])
            sequence_file.close()
            return sequence
        except ValueError:
            print('%d data in file, but reshape is in (%d, %d)' % (sequence.size, n_point, n_dim))
            continue
