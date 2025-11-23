## Data Pre-processing

### Convert from MusicXML

- Navigate to the data folder ```cd data/```
- Modify the ```ORI_FOLDER``` and ```DES_FOLDER``` in ```1_batch_xml2abc.py```, then run this script:
  ```
  python 1_batch_xml2abc.py
  ```
  This will conver the MusicXML files into standard ABC notation files.
- Modify the ```ORI_FOLDER```, ```INTERLEAVED_FOLDER```, ```AUGMENTED_FOLDER```, and ```EVAL_SPLIT``` in ```2_data_preprocess.py```:
  
  ```python
  ORI_FOLDER = ''  # Folder containing standard ABC notation files
  INTERLEAVED_FOLDER = ''   # Output interleaved ABC notation files that are compatible with CLaMP 2 to this folder
  AUGMENTED_FOLDER = ''   # On the basis of interleaved ABC, output key-augmented and rest-omitted files that are compatible with NotaGen to this folder
  EVAL_SPLIT = 0.1    # Evaluation data ratio
  ```
  then run this script:
  ```
  python 2_data_preprocess.py
  ```
  - The script will convert the standard ABC to interleaved ABC, which is compatible with CLaMP 2. The files will be under ```INTERLEAVED_FOLDER```.

  - This script will make 15 key signature folders under the ```AUGMENTED_FOLDER```, and output interleaved ABC notation files with rest bars omitted. This is the data representation that NotaGen adopts.
  
  - This script will also generate data index files for training NotaGen. It will randomly split train and eval sets according to the proportion ```EVAL_SPLIT``` defines. The index files will be named as ```{AUGMENTED_FOLDER}_train.jsonl``` and ```{AUGMENTED_FOLDER}_eval.jsonl```.

## Data Post-processing

### Preview Sheets in ABC Notation

We recommend [EasyABC](https://sourceforge.net/projects/easyabc/), a nice software for ABC Notation previewing, composing and editing.

It's needed to add a line "X:1" before each piece to present the score image in EasyABC :D

### Convert to MusicXML

- Go to the data folder ```cd data/```
- Modify the ```ORI_FOLDER``` and ```DES_FOLDER``` in ```3_batch_abc2xml.py```, then run this script:
  ```
  python 3_batch_abc2xml.py
  ```
  This will conver the standard/interleaved ABC notation files into MusicXML files.
