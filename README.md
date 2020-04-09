## Download dataset
Run a command similar to `rsync -auxvL nedc_tuh_eeg@www.isip.piconepress.com:~/data/tuh_eeg/ .`
Download dataset to `./dataset`
Details in `Instructions` part @ https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

## Processing data
`python preprocess.py`

## Training
`python main.py`

## Evaluating
`python evaluate.py --test`
one of `--train`, `--val`, `--test` is required
