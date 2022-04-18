# Installation Instructions


## Install Dependencies

1. Clone the repository:
   ```
   git clone git@github.com:meera1hahn/Graph_LED.git
   ```
2. Create a conda environment:
   ```
   cd Graph_LED
   conda create -n graphled python=3.6
   conda activate graphled
   ```
3. Install additional requirements:
   ```
   pip install -r requirements.txt
   ```

## Download WAY dataset

1. The code in this repository expects a variety of configuration and data
   files to exist in the `data` directory. The easiest way to get all of the
   required configuration files is to run the following command:

   ```
   python scripts/download-auxiliary-data.py
   ```
2. Download the Resent 152 places features for all the panos (https://github.com/peteanderson80/Matterport3DSimulator):
   ```
   cd data
   wget https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip
   unzip ResNet-152-places365.zip
   rm ResNet-152-places365.zip
   ```
3. Extract the panos features needed.
   ```
   cd ..
   python scripts/process-pano-feats.py
   ```

The first step will download WAY dataset described [here (https://meerahahn.github.io/way/data)] into the data folder.

| Dataset | Extract path | Size |
|-------------- |---------------------------- |------- |
| [way_splits.zip](https://drive.google.com/file/d/1l0qkyRjOM1VmiXYvtjPrMN9NyHgh3OXh/view) | `data/way_splits/` | 2 MB |
| [word_embeddings.zip](https://drive.google.com/file/d/1gC6Y4jqFOFkKFLSiqkt_ZGU4MM0vYIW7/view) | `data/word_embeddings/` | 13 MB |
| [floorplans.zip](https://drive.google.com/file/d/1_JHaTxty1cnZHnBKUWcNIgAPyCFx0nR7/view) | `data/floorplans/` | 103 MB |
| [connectivity.zip](https://drive.google.com/file/d/1LQ__PGY1KSNjfmGK_YqZezkSwqtdYu9c/view) | `data/connectivity/` | 1 MB |

###  Pretrained Models
We provide a trained lingUnet-skip model described in the paper for the LED task. These models are hosted on Google Drive and can be downloaded as such:

```bash
python -m pip install gdown
cd data

# LingUNet-Skip (65MB)
gdown 'https://drive.google.com/uc?id=1WTHyDEpn-4wRnvGkXCm_g7bm5_gBB8oQ'
```

## Verify the data and models directory structure

After following the steps above the `data` directory should look like this:

```
data/
  connectivity/
  distances/
  floorplans/
  geodistance_nodes.json
  models/
  node_feats/
  way_splits/
  word_embeddings/
```
