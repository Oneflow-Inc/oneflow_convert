#!/bin/bash
read_dir(){
    for file in `ls -a $1`
    do
        if [ -d $1"/"$file ]
        then
            if [[ $file != '.' && $file != '..' ]]
            then
                read_dir $1"/"$file
            fi
        else
            check_suffix $1"/"$file
        fi
    done
}

check_suffix()
{
    file=$1
    
    if [ "${file##*.}"x = "py"x ];then
        python3 -m pytest $file -v -s
        python3 /tmp/oneflow_code.py
    fi    
}
 
path="examples/x2oneflow/pytorch2oneflow/code_gen"
read_dir $path
path="examples/x2oneflow/tensorflow2oneflow/code_gen"
read_dir $path
path="examples/x2oneflow/paddle2oneflow/code_gen"
read_dir $path