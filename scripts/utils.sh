#!/bin/bash
set -e

# Make sure the result file exists and is empty
create_empty_file() {
    local result_file=${1}
    if ! [ -f ${result_file} ];
    then
        mkdir -p $(dirname ${result_file})
        touch ${result_file}
    else
        > ${result_file}
    fi
}

