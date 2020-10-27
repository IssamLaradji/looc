[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">LOOC: Localize Overlapping Objects with Count Supervision</h1>
<h5 align="center">It uses an attention mechanism to learn from the most confident regions in order to infer predictions in the less confident ones.</h5>

[![](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20LooC!%20%E2%9C%A8%20An%20Localize,%20Overlapping%20Objects%20with%20count%20any%20Supervision%20E2%9C%A8%20https://github.com/ElementAI/looc%20%F0%9F%A4%97) - [Paper link] - [Share on Facebook]
Image here


### Install requirements
`pip install -r requirements.txt` 
This command installs the Haven library which helps in managing the experiments.


## Dataset

### Trancos

- `wget http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/TRANCOS_v3.tar.gz`


## Train & Validate LOOC on Trancos

```
python trainval.py -e looc_trancos -sb <savedir_base> -d <datadir> -r 1
```
where `<datadir>` is where the data is saved (example `.tmp/data`), and  `<savedir_base>` is where the results will be saved (example `.tmp/results`)
