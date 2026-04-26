# Match-Any-Events: Zero-Shot Motion-Robust Feature Matching Across Wide Baselines for Event Cameras

**Official implementation for the paper: [Match-Any-Events: Zero-Shot Motion-Robust Feature Matching Across Wide Baselines for Event Cameras](TODO)**

<img width="2007" height="782" alt="teaser" src="https://github.com/user-attachments/assets/2402f354-def2-4f4a-bedc-164448bd34c3" />

## TODO (Updated 4/20)
- [x] ~~Realtime Demo~~
- [x] ~~Evaluation Script~~
- [ ] Pretrained Weights (~~Event matching~~ | Cross-modality matching)
- [ ] Data generation pipeline
- [ ] Training Script
- [ ] SETS Module Script

## Requirements

Code was tested with the following environment setup. 

* **System:** Ubuntu 20.04 / 22.04
* **Python:** 3.10
* **CUDA:** 12.9

Install requirements as in the requirements.txt file.

```bash
pip install -r requirements.txt
```
### Dataset
Coming Soon!

### Pretrained Weights on E-Megadepth
Pretrained MatchAnyEvents model weights can be downloaded [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rzhan158_jh_edu/IQDfqv8vdjGkRYBfKr9ShVzJAYs-hDzi-0y9lN7vI-9kmqw?e=szt9Od).

Create a folder named `pretrained` and move the pretrained weights into this folder so that loading is supported directly in the demo and evaluation scripts.

```text
Match-Any-Events/
├── data/
│   ├── ECM/
│   └── E-MegaDepth/
└── pretrained/
    └── weights.pth
```

## Realtime Demo
https://github.com/user-attachments/assets/499cae1e-2d56-4894-91d4-0e6ed692b2d2

* **System:** Ubuntu 22.04
* **Cameras:** 2 x Prophesee Gen3 VGA (640x480)
* **SDK:** Metavision SDK 4.6 [https://docs.prophesee.ai/stable/installation/index.html](https://docs.prophesee.ai/stable/installation/index.html)

Run the demo script using serial numbers of your cameras.

```bash
python realtime_demo.py --s1 YOUR_SERIAL_NUMBER1 --s2 YOUR_SERIAL_NUMBER2
```
## Evaluation
Coming Soon!

## Citation
```bibtex
@article{zhang2026match,
  title={Match-Any-Events: Zero-Shot Motion-Robust Feature Matching Across Wide Baselines for Event Cameras},
  author={Zhang, Ruijun and Su, Hang and Daniilidis, Kostas and Wang, Ziyun},
  journal={arXiv preprint arXiv:2604.18744},
  year={2026}
}
```

## Acknowledgements

This project is built upon the foundational code provided by [ELoFTR: https://github.com/zju3dv/efficientloftr](https://github.com/zju3dv/efficientloftr). We thank the authors for releasing their code.

We would like to thank the authors for their open-source implementations, which inspired parts of this project:
* [ESim: https://github.com/uzh-rpg/rpg_esim](https://github.com/uzh-rpg/rpg_esim)
* [A-ViT: https://github.com/NVlabs/A-ViT](https://github.com/NVlabs/A-ViT)

