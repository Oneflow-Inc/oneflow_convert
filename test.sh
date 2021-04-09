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
        pytest $file
    fi    
}
 
path="examples"
read_dir $path