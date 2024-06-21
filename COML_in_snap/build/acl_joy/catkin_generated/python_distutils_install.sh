#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/src/acl_joy"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/install/lib/python3/dist-packages:/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/build/acl_joy/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/build/acl_joy" \
    "/usr/bin/python3" \
    "/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/src/acl_joy/setup.py" \
     \
    build --build-base "/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/build/acl_joy" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/install" --install-scripts="/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/install/bin"
