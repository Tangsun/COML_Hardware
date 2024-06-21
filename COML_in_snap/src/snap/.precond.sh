#!/bin/bash

# This bash script is to be run in the mitacl/sfpro-board Docker image
# before catkin_make to satisfy all the necessary dependencies and
# preconditions.
#
# Note that the original timestamps are preserved to prevent unnecessary
# recompilation.

# dev env workspace directory (where dependencies have been built)
WSDIR=$1
CPU=$2

cp $WSDIR/esc_interface/build/$CPU/libesc_interface.so /usr/lib/
touch -r $WSDIR/esc_interface/build/$CPU/libesc_interface.so /usr/lib/libesc_interface.so

ROOT=$WSDIR/esc_interface/include/esc_interface
TRGT=/usr/include
BASE=
cp -r $ROOT $TRGT
find $ROOT | while read f; do
    if [[ -z $BASE ]]; then
        BASE=$(basename $f)
    fi

    # get new filename path (in local target directory)
    subf=$(echo $f | sed s~$ROOT~$TRGT/$BASE~g)

    touch -r $f $subf
done

# snap_apm library
if [[ -f $WSDIR/snap_apm/build/$CPU/libsnap_apm.so ]]; then

    cp $WSDIR/snap_apm/build/$CPU/libsnap_apm.so /usr/lib/
    touch -r $WSDIR/snap_apm/build/$CPU/libsnap_apm.so /usr/lib/libsnap_apm.so

    ROOT=$WSDIR/snap_apm/include/snap_apm
    TRGT=/usr/include
    BASE=
    cp -r $ROOT $TRGT
    find $ROOT | while read f; do
        if [[ -z $BASE ]]; then
            BASE=$(basename $f)
        fi

        # get new filename path (in local target directory)
        subf=$(echo $f | sed s~$ROOT~$TRGT/$BASE~g)

        touch -r $f $subf
    done

fi
