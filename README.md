This repository uses [labelImg](https://github.com/HumanSignal/labelImg) to annotate bounding boxes on images for object detection and person detection.
The resulting ground truth annotations are then used to generated a PDF comparing the annotations and detected bounded boxes along with the IoU.
The detected bounding boxes and images are retrieved from the results returned by robots using the [metrics_refbox](https://github.com/HEART-MET/metrics_refbox) during the [HEART-MET competitions](https://metricsproject.eu/healthcare/).
For person detection images, faces of people are blurred using the face detection models found [here](https://github.com/cs-giung/face-detection-pytorch).

## Dependencies
### LabelImg
Clone [labelImg](https://github.com/HEART-MET/labelImg) on your system and build it:
```
sudo apt-get install pyqt5-dev-tools
git clone git@github.com:HEART-MET/labelImg.git ~/labelImg
cd ~/labelImg && make qt5py3
```
### Face detection
If you're using face detection, pytorch and torchvision are required:
```
pip3 install torch torchvision
```
Download the weights for the TinyFace model from this [Google Drive link](https://drive.google.com/file/d/12wJtproN2cPFlUmff_c8AdkmMKEu03yc/view?usp=sharing) originally obtained from [here](https://github.com/cs-giung/face-detection-pytorch), and save it to `./detectors/tinyface/weights/checkpoint_50.pth`.

### Texlive

If you want to generate the report, texlive is required
```
sudo apt install texlive-full
```


## Usage

```
python annotate.py <path to results>
```

### Options
```
[--no-report] # don't generate a PDF report (default will generate a report)
[--labelImg_path path_to_labelImg.py] # full path to labelImg.py (default: ~/labelImg/labelImg.py)
[--report_name name_of_report] # name of PDF file (default: root)
```
