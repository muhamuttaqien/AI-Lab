mkdir datasets

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./datasets/
wget http://images.cocodataset.org/zips/train2014.zip -P ./datasets/
wget http://images.cocodataset.org/zips/val2014.zip -P ./datasets/

unzip ./datasets/captions_train-val2014.zip -d ./datasets/
rm ./datasets/captions_train-val2014.zip

unzip ./datasets/train2014.zip -d ./datasets/
rm ./datasets/train2014.zip

unzip ./datasets/val2014.zip -d ./datasets/
rm ./datasets/val2014.zip
