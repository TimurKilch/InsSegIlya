#!/bin/bash

/etc/init.d/fcgiwrap start
chmod 766 /var/run/fcgiwrap.socket
cd /gitfolder/instance_seg_abris && git restore . && git pull
rm -rf .git
chmod +x /gitfolder/instance_seg_abris/main.py
nginx -g "daemon off;"