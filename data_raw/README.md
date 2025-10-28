under project directory

## Download raw datasets 

### FVQA 
- [Question+Answer+Rational](https://github.com/wangpengnorman/FVQA?tab=readme-ov-file): download to ./data_raw

- COCO 2017 images
```bash
export COCO_DIR=./data/images/aokvqa/
mkdir -p ${COCO_DIR}

for split in train val test; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}; rm "${split}2017.zip"
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip
```


### A-OKVQA
- [QuestionAnswerRational](https://github.com/allenai/aokvqa): download to ./data_raw
- COCO 2014 & IMAGENET images: 
copy new_dataset_release/images to data/images/fvqa


## Pre-engineer notebooks
- data_raw/process_fvqa.ipynb
- data_raw/process_aokvqa.ipynb


