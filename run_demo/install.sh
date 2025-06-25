#!/bin/bash - 

set -euo pipefail
stage=0
venv_dir=anaconda
installer=Miniconda3-py310_24.11.1-0-Linux-x86_64.sh

#source /mnt/asr_hot/bin/linux/pkg/lmod/enable
#module load cuda/11.7 || exit 1

if [ $stage -le 0 ] ; then
    echo "$0: Stage 0: Download anaconda"
    if [ ! -f $venv_dir/.done ]; then
        echo "Downloading anaconda installer"
        [ ! -f $installer ] && \
            wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 20 \
                https://repo.anaconda.com/miniconda/$installer
        echo "Installing anaconda"
        bash $installer -b -p $venv_dir
        touch $venv_dir/.done
    fi
fi
source $venv_dir/bin/activate || exit 1
echo "Installing pip"
conda install -y pip
if [ $stage -le 1 ] ; then 
    echo "Installing base requirements"
    pip install -r ./base_requirements.txt
    pip install git+https://nid-gitlab.ad.speechpro.com/asr2/inex.git
fi


exit 1