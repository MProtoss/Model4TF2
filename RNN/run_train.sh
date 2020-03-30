#sh
datapath="/search/odin/lizhengyu/code2020/RNN/data/"
emdfile=$datapath"emd_table.tfrecord"
tfrecordfile=$datapath"star_classfy.tfrecord"

python simple_rnn_train.py $emdfile $tfrecordfile
