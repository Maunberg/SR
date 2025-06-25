set -eou pipefail

source anaconda/bin/activate

echo "Setting up pl environment"
# Check if pl environment exists
if ! conda env list | grep -q "^pl "; then
    echo "Creating new pl environment..."
    conda create -y --name pl python=3.10 # 3.12 doesn't work with fairseq
    conda activate pl
    # pip to 24.0 because 25 will fail fairseq installation
    # faiseq requires omegaconf > 2.0.5 < 2.1.0 but 
    # installation of 2.0.6 breaks in pip >= 24.1
    # "DEPRECATION: omegaconf 2.0.6 has a non-standard dependency specifier PyYAML>=5.1.*. pip 24.1 will..."
    conda install pip=24.0
    echo "Conda env created"
else
    echo "'pl' environment already exists"
fi

conda activate pl

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install pandas matplotlib
pip install 'transformers[torch]'
pip install lightning 
pip install ipython jupyter
#pip install git+https://github.com/medbar/webdataset.git
pip install webdataset
pip install traceback-with-variables
#pip install git+https://nid-gitlab.ad.speechpro.com/asr2/inex.git  # this is custom env for inference only
pip install protobuf==3.20.*
pip install sentencepiece
pip install soundfile
pip install peft
pip install torchmetrics

#pip install speechbrain

pip install omegaconf==2.0.6 # version for fairseq


# omegaconf is installed to 2.3.0 with because of inex 