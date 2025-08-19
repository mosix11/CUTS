# https://github.com/mlfoundations/task_vectors/issues/1#issuecomment-2236459857
sudo apt-get install p7zip-full
mkdir data
cd data
# https://www.kaggle.com/docs/api
# put your credentials here or put kaggle.json in ~/.kaggle/kaggle.josn
# export KAGGLE_USERNAME=<your username>
# export KAGGLE_KEY=<your kaggle key>

# stanford cars dataset (ref: https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616)
mkdir StandfordCars && cd StandfordCars
kaggle datasets download -d jessicali9530/stanford-cars-dataset
kaggle datasets download -d abdelrahmant11/standford-cars-dataset-meta
7z x standford-cars-dataset-meta.zip
7z x stanford-cars-dataset.zip
tar -xvzf car_devkit.tgz
mv cars_test a
mv a/cars_test/ cars_test
rm -rf a
mv cars_train a
mv a/cars_train/ cars_train
rm -rf a
mv 'cars_test_annos_withlabels (1).mat' cars_test_annos_withlabels.mat
rm -rf 'cars_annos (2).mat' *.zip
cd ..

# ressic45
mkdir Resisc45 && cd Resisc45
kaggle datasets download -d aqibrehmanpirzada/nwpuresisc45
7z x NWPU-RESISC45.rar
wget -O resisc45-train.txt "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt"
wget -O resisc45-val.txt "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt"
wget -O resisc45-test.txt "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt"
rm -rf NWPU-RESISC45.rar
cd ..

# dtd
mkdir DTD && cd DTD
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm -rf dtd-r1.0.1.tar.gz
mv dtd/images images
mv dtd/imdb/ imdb
mv dtd/labels labels
cat labels/train1.txt labels/val1.txt > labels/train.txt
cat labels/test1.txt > labels/test.txt

# euro_sat
mkdir EuroSAT && cd EuroSAT
wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip
7z x EuroSAT.zip
rm -rf EuroSAT.zip

# sun397
mkdir Sun397 && cd Sun397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
7z x Partitions.zip
tar -xvzf SUN397.tar.gz
rm -rf SUN397.tar.gz