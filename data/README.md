## Setting up the data

The benchmarking scripts expects all of the data files to be present in `data/atis-2/` directory.

```
mkdir data/atis-2/
```

To setup the data for benchmarking under these requirements, do the following:

1. Download all of the files from https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/atis-2 sand save them into the `atis-2` directory.
> *Please see this data set's applicable license for terms and conditions. Intel does not own the rights to this data set and does not confer any rights to it.*


	
```
cd data/atis-2/

wget  https://raw.githubusercontent.com/sz128/slot_filling_and_intent_detection_of_SLU/master/data/atis-2/train
wget https://raw.githubusercontent.com/sz128/slot_filling_and_intent_detection_of_SLU/master/data/atis-2/test
wget https://raw.githubusercontent.com/sz128/slot_filling_and_intent_detection_of_SLU/master/data/atis-2/valid
wget https://raw.githubusercontent.com/sz128/slot_filling_and_intent_detection_of_SLU/master/data/atis-2/vocab.intent
wget https://raw.githubusercontent.com/sz128/slot_filling_and_intent_detection_of_SLU/master/data/atis-2/vocab.slot
```
	
2. Combine the `atis-2/train` and `atis-2/valid` files into one called `atis-2/train_all`.  In Linux, this can be done from the current directory using

```shell
cat train valid > train_all
cd ../..
```
