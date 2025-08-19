# https://github.com/mlfoundations/task_vectors/issues/1#issuecomment-2236459857
sudo apt-get install p7zip-full
mkdir data; cd data
# https://www.kaggle.com/docs/api
# put your credentials here or put kaggle.json in ~/.kaggle/kaggle.josn
# export KAGGLE_USERNAME=<your username>
# export KAGGLE_KEY=<your kaggle key>

mkdir StanfordCars; cd StanfordCars
# one other option emanuelriquelmem/stanford-cars-pytorch
kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars
7z x stanford-cars-pytorch.zip
rm stanford-cars-pytorch.zip
