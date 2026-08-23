// STrace Inspector
// Copyright (C) 2024 Forschungszentrum Juelich GmbH, Juelich Supercomputing Centre

// Contributors:
// - Aravind Sankaran

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

function makeDtFilter(selector) {
    return function (settings, data, dataIndex) {
        if (settings.nTable !== $(selector)[0]) {
            return true; // not the target table
        }
        var inputs = $(selector).find('thead input');

        for (var i = 0; i < inputs.length; i++) {
            var input = $(inputs[i]);

            var colIdx = input.data('col');
            if (colIdx === undefined || isNaN(colIdx)) {
                continue;
            }

            var rawVal = input.val().toLowerCase().trimStart();
            if (!rawVal) continue; // skip empty filters

            // Each data corresponds to a row. Each index of data corresponds to a column
            var cellData = (data[colIdx] || '').toString().toLowerCase(); // cell value

            // Split by OR (|)
            var orGroups = rawVal.split('|').map(v => v.trimStart());
            var matchedAny = false;

            // For each OR group (each can contain multiple AND conditions)
            for (var g = 0; g < orGroups.length; g++) {
                var andFilters = orGroups[g].split('&').map(v => v.trim());
                var matchedAll = true;

                for (var j = 0; j < andFilters.length; j++) {
                    var filter = andFilters[j];
                    if (!filter) continue;

                    var match = false;

                    // --- Text match ---
                    if (!filter.startsWith('>') && !filter.startsWith('<') && !filter.startsWith('=')) {
                        if (cellData.includes(filter)) 
                            match = true;
                    }

                    // --- Numeric comparisons ---
                    var cellNum = parseFloat(cellData);
                    var num = NaN;

                    if (filter.startsWith('>=')) {
                        num = parseFloat(filter.substring(2));
                        if (!isNaN(cellNum) && !isNaN(num) && cellNum >= num) match = true;
                    } else if (filter.startsWith('<=')) {
                        num = parseFloat(filter.substring(2));
                        if (!isNaN(cellNum) && !isNaN(num) && cellNum <= num) match = true;
                    } else if (filter.startsWith('>')) {
                        num = parseFloat(filter.substring(1));
                        if (!isNaN(cellNum) && !isNaN(num) && cellNum > num) match = true;
                    } else if (filter.startsWith('<')) {
                        num = parseFloat(filter.substring(1));
                        if (!isNaN(cellNum) && !isNaN(num) && cellNum < num) match = true;
                    } else if (filter.startsWith('=')) {
                        num = parseFloat(filter.substring(1));
                        if (!isNaN(cellNum) && !isNaN(num) && cellNum == num) match = true;
                    }

                    // If any AND condition fails → group fails
                    if (!match) {
                        matchedAll = false;
                        break;
                    }
                }

                // If one OR group passes → overall passes
                if (matchedAll) {
                    matchedAny = true;
                    break;
                }
            }

            // If none of the OR groups matched for this column, reject row
            if (!matchedAny) return false;
        }

        return true;
    }
}