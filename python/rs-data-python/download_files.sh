#!/bin/bash

# $1 code: ml-1m, bx, etc
# $2 url: htttp://blabla.zip
process_ds () {
    code=${1%;*}
    url=${1#*;}
    echo "CODE: $code"
    echo "URL: $url"
    if [ ! -d "$code" ]; then
        echo "Downloading..."
        mkdir $code
        cd $code
        wget $url -O $code.zip
        unzip $code.zip -d .
        cd ..
    else
        echo "Nothing to do."
    fi
}

# Book-Crossing Dataset
# http://www2.informatik.uni-freiburg.de/~cziegler/BX/

DS="
bx;http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
ml-1m;https://files.grouplens.org/datasets/movielens/ml-1m.zip
"

mkdir rawdata
cd rawdata

#for data in $DS; do
    #process_ds $data
#done

#exit


if [ ! -d "bx" ]; then
    # Book-Crossing Dataset
    # url: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
    mkdir bx
    cd bx
    wget http://www2.informatik.uni-freiburg.de/\~cziegler/BX/BX-CSV-Dump.zip -O bx.zip
    unzip bx.zip
    cd ..
fi

if [ ! -d "ml-1m" ]; then
    # Book-Crossing Dataset
    # url: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
    mkdir ml-1m
    cd ml-1m
    wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O ml-1m.zip
    unzip ml-1m.zip
    cd ..
fi
