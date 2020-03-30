#sh
infile="./emd_table.txt"
tfrecordfile="./emd_table.tfrecord"

python EmbdTFRecordMaker.py $infile $tfrecordfile
