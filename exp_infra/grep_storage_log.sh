# !/bin/bash

docker logs modyn-storage-1 &> ~/tmp.txt
grep '\[JZ\]' ~/tmp.txt