#sh
infile="./rnn_random_data.txt"
#infile="../Tools/data.txt"
tfrecordfile="./rnn_test.tfrecord"

python TFRecordMaker.py $infile $tfrecordfile
