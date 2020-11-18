# AUTOMATIC-REGION-SELECTION-FOR-OBJECTIVE-SHARPNESS-ASSESSMENT-OF-MOBILE-DEVICE-PHOTOS
The framework of our local region selection algorithm.

![image](https://github.com/superuct/AUTOMATIC-REGION-SELECTION-FOR-OBJECTIVE-SHARPNESS-ASSESSMENT-OF-MOBILE-DEVICE-PHOTOS/blob/main/figures/autoRegionS.pdf)

## Usage

Assign the input path to rootPath, like "pic".

- pic/001/A_001.jpg
- pic/001/B_001.jpg
- ...
- pic/001/O_001.jpg
- pic/002/A_002.jpg
- pic/002/B_002.jpg
- ...
- pic/002/O_002.jpg
- ...
- pic/100/O_100.jpg


Assign the output path to outpath.

Run main program.

    python mainSelectionWithoutSalient&Depth.py
    python mainSelection.py
    
## Citation

If you use this code or pre-trained models, please cite the following:

    @inproceedings{lu2020automatic,
      title={Automatic Region Selection For Objective Sharpness Assessment Of Mobile Device Photos},
      author={Lu, Qiang and Zhai, Guangtao and Zhu, Wenhan and Zhu, Yucheng and Min, Xiongkuo and Zhang, Xiao-Ping and Yang, Hua},
      booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
      pages={106--110},
      year={2020},
      organization={IEEE}
    }
