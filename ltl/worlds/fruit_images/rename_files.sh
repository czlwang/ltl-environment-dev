#!/bin/bash
find . -type f | while read FILE ; do
    newfile="$(echo ${FILE} |sed -e 's/apple/gem/g' -e 's/orange/gold/g' -e 's/pear/iron/g' -e 's/flag/factory/g' -e 's/tree/tree/g' -e 's/house/workbench/g')" ;
    mv "${FILE}" "${newfile}" ;
done 
