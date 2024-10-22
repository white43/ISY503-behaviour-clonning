# Udacity Self-Driving Car Simulator

## Instructions

1. Clone this repository `git clone https://github.com/white43/ISY503-behavioral-cloning.git`
2. `cd ISY503-behavioral-cloning`
3. Download datasets (see below)
4. Unpack `Track-1.zip` and `Track-2.zip` to `Track-1` and `Track-2` directories 
5. Basic check that everything is correct so far: `ls -l Track-{1,2}/f1/driving_log.csv` should list two files
6. `pip install -r requirements.txt`

## Datasets

If you're a human, remove 32-character-long random string and one dot from the file name and download it.

1. Track 1. `https://torrens-files.s3.ap-southeast-2.amazonaws.com/ISY503/Track-1.94tpDvQ9gK6I5jMuX3lA2z5uG4Co4kIE.zip`
2. Track 2. `https://torrens-files.s3.ap-southeast-2.amazonaws.com/ISY503/Track-2.Wf9ePu1A8tK3Nn2j1Y17JzUVH3cNqqh8.zip`

## Training

Run the following command to train your model to drive on Track 1:

```commandline
python model.py --sources Track-1/f1 Track-2/b1 Track-1/fa1 Track-2/ba1 --train-on-autonomous-center
```

And the following command is for Track 2:

```commandline
python model.py --sources Track-2/f1 Track-2/f2 Track-2/fa1 Track-2/b1 Track-2/b2 Track-2/ba1 --train-on-autonomous-center
```

## Driving

Download model via the link `https://torrens-files.s3.ap-southeast-2.amazonaws.com/ISY503/model.keras` and save it 
to the project directory. Then run the below command.

```commandline
python drive.py --file model.keras
```

## Gathering autonomous data for training

1. Create a new folder for the data to be gathered
2. Start driving with a special command line argument `--save-image-to`

```commandline
python drive.py --file model.keras --save-image-to path/to/directory
```

## Training on gathered autonomous data

Autonomous data contains only center images. So, it is mandatory to provide some human-gathered data with left and right 
images to instruct the car how to get back when it diverges from the center of the road.

```commandline
python model.py --sources [Track-N/f1, ...] Track-N/fa1 Track-N/ba1 --train-on-autonomous-center
```
