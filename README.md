# HI-MIA Crash Course Homework

# 0. Abstract

This repository is slightly modified based on [the FFSVC2022 baseline system](https://github.com/FFSVC/FFSVC2022_Baseline_System), including:

- Data preparation
- Model training
- Embedding extracting
- Performance calculating

Please visit https://ffsvc.github.io/ for more information about the challenge.

The code has been tested using the following environment:

- **Linux OS: Ubuntu 20.04.5 LTS, 64-bit**
- **Graphics: NVIDIA GeForce RTX 3080 Laptop**

The epitome of the most significant ideas are in `DURUII` , including: nn.Conv1d, Feature Augmentation, Model.

# 1. System Introduction

The system adopts the **online data augmentation** method for model training. Please prepare the <a href="https://www.openslr.org/17/">MUSAN </a> and <a href="https://www.openslr.org/28/">RIR_NOISES </a> dataset ([HI-MIA](https://www.openslr.org/85/) and [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/), needless to say, are also needed) and modify the path of `./data/MUSAN/` and `./data/RIR_Noise/` files as your data path. The acoustic feature extraction and data augmentation depend on the torchaudio package, whose backend is sox.

P.S. The command line of concatenating parts of a zip file might not work the same on Windowns platform, which had nearly drove me crazy before I installed Ubuntu System.

## Training

The training config is saved in `./config/*` files, and the training log is saved in `exp/PATH_SAVE_DIR`.

## Testing

There are three modes for `scoring.py`,

1. Extract speaker embedding and compute the EER and mDCF

```python
scoring = True
onlyscoring = False
```

2. Extract speaker embedding

```python
scoring = False
onlyscoring = False
```

3. Compute EER and mDCF

```python
scoring = False/True
onlyscoring = True
```

Please set the test mode in `./exp/config_scoring.py` before running the `scoring.py`

# 2. System Pipeline

The system adopts the `pre-train + finetuning` strategy. First, the Vox2dev data is used to train the pre-trained model. Then, Vox2dev and HI-MIA data are integrated (or mixed) to finetuning the pre-trained model.

## Step 1. Data preparation

The data preparation file follows the Kaldi form that participants need `wav.scp`, `utt2spk` and `spk2utt` files for training dir, and `wav.scp` and `trials` for valuation dir.

The `./data/vox2dev/` shows the training example files and `./data/vox1test` shows the valuation example files. There are some data dir need to be prepared in the baseline system recipe:

<pre><font color="#729FCF"><b>./data</b></font>
├── <font color="#729FCF"><b>ftmix</b></font>
│   ├── spk2utt
│   ├── utt2spk
│   └── wav.scp
├── <font color="#729FCF"><b>himia_eval</b></font>
│   ├── trial_mic
│   ├── <font color="#34E2E2"><b>trials</b></font> -&gt; trial_mic
│   ├── wav_list
│   └── wav.scp
├── <font color="#729FCF"><b>himia_train</b></font>
│   ├── spk2utt
│   ├── utt2spk
│   ├── wav_list
│   └── wav.scp
├── <font color="#729FCF"><b>MUSAN</b></font>
│   ├── music_wav_list
│   └── noise_wav_list
├── <font color="#729FCF"><b>RIR_Noise</b></font>
│   └── rir_list
├── <font color="#729FCF"><b>vox1test</b></font>
│   ├── trials
│   └── wav.scp
└── <font color="#729FCF"><b>vox2dev</b></font>
    ├── <font color="#8AE234"><b>spk2utt</b></font>
    ├── <font color="#8AE234"><b>utt2spk</b></font>
    └── <font color="#8AE234"><b>wav.scp</b></font>
</pre>

### RIR_Noise and MUSAN

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ pwd
/home/durui/Documents/FFSVC2022_Baseline_System
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ cd data/MUSAN/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/MUSAN</b></font>$ ls -lh /media/durui/DATA/musan/
total 256K
drwxr-xr-x 7 durui durui  64K Oct 30  2015 <font color="#729FCF"><b>music</b></font>
drwxr-xr-x 4 durui durui  64K Oct 30  2015 <font color="#729FCF"><b>noise</b></font>
-rwxr-xr-x 1 durui durui 1.8K Oct 30  2015 <font color="#8AE234"><b>README</b></font>
drwxr-xr-x 4 durui durui  64K Oct 30  2015 <font color="#729FCF"><b>speech</b></font>
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/MUSAN</b></font>$ find /media/durui/DATA/musan/music -name &quot;*.wav&quot; &gt; music_wav_list</pre>

The find-command hold true of the rest of MUSAN and RIR_Noise data preparation.

### VoxCeleb

Once you have sucessfully downloaded and unzipped audio dataset following the instructions on https://mm.kaist.ac.kr/datasets/voxceleb/, you can replace the PATH into your own with ease.

### HI-MIA

#### Training

##### wav_list

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ pwd
/home/durui/Documents/FFSVC2022_Baseline_System
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ cd data
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ mkdir himia_train himia_eval
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ cd himia_train/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ find /media/durui/DATA/HIMIA/train/ -name &quot;*.wav&quot; &gt; wav_list
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ wc -l wav_list 
993083 wav_list
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ head wav_list 
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_10_N2631.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_2_10_S0056.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_16_N1779.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_04_N3361.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_09_F3787.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_07_S3657.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_11_S1409.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_11_N1433.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_16_S1801.wav
/media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_06_F2278.wav</pre>

##### wav.scp

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ awk -F &apos;/&apos; &apos;{print $NF,$0}&apos; wav_list | sed &apos;s/\.wav//1&apos; &gt; wav.scp
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ head wav.scp 
SV0219_6_10_N2631 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_10_N2631.wav
SV0219_2_10_S0056 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_2_10_S0056.wav
SV0219_5_16_N1779 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_16_N1779.wav
SV0219_6_04_N3361 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_04_N3361.wav
SV0219_6_09_F3787 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_09_F3787.wav
SV0219_6_07_S3657 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_6_07_S3657.wav
SV0219_5_11_S1409 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_11_S1409.wav
SV0219_5_11_N1433 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_11_N1433.wav
SV0219_5_16_S1801 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_16_S1801.wav
SV0219_5_06_F2278 /media/durui/DATA/HIMIA/train/SPEECHDATA/wav/SV0219/SV0219_5_06_F2278.wav
</pre>

##### utt2spk and spk2utt

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ awk -F &apos;/&apos; &apos;{print $NF,$(NF-1)}&apos; wav_list | sed &apos;s/\.wav//1&apos; &gt; utt2spk
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ head utt2spk 
SV0219_6_10_N2631 SV0219
SV0219_2_10_S0056 SV0219
SV0219_5_16_N1779 SV0219
SV0219_6_04_N3361 SV0219
SV0219_6_09_F3787 SV0219
SV0219_6_07_S3657 SV0219
SV0219_5_11_S1409 SV0219
SV0219_5_11_N1433 SV0219
SV0219_5_16_S1801 SV0219
SV0219_5_06_F2278 SV0219
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ ../../tools/utt2spk_to_spk2utt.pl &lt; utt2spk &gt; spk2utt
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_train</b></font>$ wc -l *
      254 spk2utt
   993083 utt2spk
   993083 wav_list
   993083 wav.scp
  2979503 total</pre>

#### Evaluation

##### wav.scp

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ cd himia_eval/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ pwd
/home/durui/Documents/FFSVC2022_Baseline_System/data/himia_eval
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ find /media/durui/DATA/HIMIA/test_v2/  -name &quot;*.wav&quot; &gt; wav_list
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ awk -F &apos;/&apos; &apos;{print $NF,$0}&apos; wav_list | sed &apos;s/\.wav//1&apos; &gt; wav.scp
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ head wav.scp 
SV0297_2_00_F0041 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0041.wav
SV0297_2_00_F0042 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0042.wav
SV0297_2_00_F0043 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0043.wav
SV0297_2_00_F0044 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0044.wav
SV0297_2_00_F0045 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0045.wav
SV0297_2_00_F0046 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0046.wav
SV0297_2_00_F0047 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0047.wav
SV0297_2_00_F0048 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0048.wav
SV0297_2_00_F0049 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0049.wav
SV0297_2_00_F0050 /media/durui/DATA/HIMIA/test_v2/wav/SV0297_2_00_F0050.wav</pre>

##### trials

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ sed &apos;s/nontarget/0/&apos; /media/durui/DATA/HIMIA/test_v2/trials_mic | sed &apos;s/target/1/&apos; &gt; trial_mic
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ head trial_mic
SV0299_7_01_S0096.wav SV0333_2_{}_N0124.wav 0
SV0304_7_01_N0002.wav SV0335_5_{}_S0092.wav 0
SV0340_7_01_S0097.wav SV0340_2_{}_F0058.wav 1
SV0330_7_01_F0058.wav SV0317_2_{}_N0001.wav 0
SV0302_7_01_S0097.wav SV0304_5_{}_S0092.wav 0
SV0336_7_01_F0049.wav SV0336_5_{}_N0121.wav 1
SV0321_7_01_F0043.wav SV0321_2_{}_N0128.wav 1
SV0302_7_01_S0085.wav SV0302_6_{}_N0019.wav 1
SV0300_7_01_N0138.wav SV0337_2_{}_N0010.wav 0
SV0313_7_01_N0128.wav SV0312_2_{}_S0093.wav 0
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ grep SV0299_7_01_S0096.wav wav.scp 
SV0299_7_01_S0096 /media/durui/DATA/HIMIA/test_v2/wav/<font color="#EF2929"><b>SV0299_7_01_S0096.wav</b></font>
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ grep SV0333_2_.*_N0124.wav wav.scp
<font color="#EF2929"><b>SV0333_2_00_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_00_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_01_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_01_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_02_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_02_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_03_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_03_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_04_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_04_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_05_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_05_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_06_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_06_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_07_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_07_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_08_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_08_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_09_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_09_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_10_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_10_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_11_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_11_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_12_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_12_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_13_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_13_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_14_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_14_N0124.wav</b></font>
<font color="#EF2929"><b>SV0333_2_15_N0124 /media/durui/DATA/HIMIA/test_v2/wav/SV0333_2_15_N0124.wav</b></font>
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ grep SV0333_2_.*_N0124.wav wav.scp | wc -l
16
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ 
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ 
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ 
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ 
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/himia_eval</b></font>$ ln -s trial_mic trials</pre>

### Fine-tune Mix

##### wav.scp && spk2utt & utt2spk

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ pwd
/home/durui/Documents/FFSVC2022_Baseline_System
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ cd ./data/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ mkdir ftmix
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ cat vox2dev/wav.scp himia_train/wav.scp &gt; ftmix/wav.scp
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ cat vox2dev/utt2spk himia_train/utt2spk &gt; ftmix/utt2spk
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data</b></font>$ cd ftmix/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/data/ftmix</b></font>$ ../../tools/utt2spk_to_spk2utt.pl &lt; utt2spk &gt; spk2utt</pre>

##### model

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ cd exp/Vox2dev_80FBANK_ECAPATDNN_AAMsoftmax_256_dist/ftmix/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/exp/Vox2dev_80FBANK_ECAPATDNN_AAMsoftmax_256_dist/ftmix</b></font>$ ln -s ../model_43.pkl ./
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/exp/Vox2dev_80FBANK_ECAPATDNN_AAMsoftmax_256_dist/ftmix</b></font>$ ls
<font color="#34E2E2"><b>model_43.pkl</b></font></pre>

## Step 2. Training Close-talking model (training with Vox2dev data)

Modify the parameters in `./config/config_ecapatdnn_dist.py` before training. The default model is ECAPA-TDNN. Training with DDP is highly recommended.

```shell
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 train_dist.py &
```

To peep the log after trainning, you can use:

<pre>(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ pwd
/home/durui/Documents/FFSVC2022_Baseline_System
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System</b></font>$ cd exp/Vox2dev_80FBANK_ECAPATDNN_AAMsoftmax_256_dist/
(base) <font color="#8AE234"><b>durui@Zephyrus</b></font>:<font color="#729FCF"><b>~/Documents/FFSVC2022_Baseline_System/exp/Vox2dev_80FBANK_ECAPATDNN_AAMsoftmax_256_dist</b></font>$ cat train.out | grep EER
Epoch 0	  lr 0.005000	  <font color="#EF2929"><b>EER</b></font> 5.4239	  cost 0.4968
Epoch 1	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 5.3335	  cost 0.4705
Epoch 2	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 5.3814	  cost 0.4740
Epoch 3	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.6157	  cost 0.4312
Epoch 4	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.6529	  cost 0.4779
Epoch 5	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.6210	  cost 0.4082
Epoch 6	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.5146	  cost 0.4421
Epoch 7	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.4668	  cost 0.4367
Epoch 8	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.3551	  cost 0.4041
Epoch 9	  lr 0.010000	  <font color="#EF2929"><b>EER</b></font> 4.6848	  cost 0.4182
Epoch 10	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.3823	  cost 0.2233
Epoch 11	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.2547	  cost 0.2110
Epoch 12	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.3557	  cost 0.2080
Epoch 13	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.3557	  cost 0.2252
Epoch 14	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.2068	  cost 0.2087
Epoch 15	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.0260	  cost 0.1896
Epoch 16	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.1536	  cost 0.2264
Epoch 17	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.2600	  cost 0.2282
Epoch 18	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.1962	  cost 0.2141
Epoch 19	  lr 0.001000	  <font color="#EF2929"><b>EER</b></font> 2.0685	  cost 0.2047
Epoch 20	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.6485	  cost 0.1595
Epoch 21	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.5687	  cost 0.1673
Epoch 22	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4570	  cost 0.1592
Epoch 23	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.5102	  cost 0.1545
Epoch 24	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4464	  cost 0.1507
Epoch 25	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4783	  cost 0.1527
Epoch 26	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4411	  cost 0.1552
Epoch 27	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4198	  cost 0.1490
Epoch 28	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4198	  cost 0.1428
Epoch 29	  lr 0.000100	  <font color="#EF2929"><b>EER</b></font> 1.4304	  cost 0.1530
Epoch 30	  lr 0.000010	  <font color="#EF2929"><b>EER</b></font> 1.3134	  cost 0.1281
Epoch 31	  lr 0.000010	  <font color="#EF2929"><b>EER</b></font> 1.3666	  cost 0.1348
Epoch 32	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.2975	  cost 0.1376
Epoch 33	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3347	  cost 0.1477
Epoch 34	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3613	  cost 0.1402
Epoch 35	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3879	  cost 0.1441
Epoch 36	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.2762	  cost 0.1498
Epoch 37	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3613	  cost 0.1375
Epoch 38	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3773	  cost 0.1418
Epoch 39	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3719	  cost 0.1526
Epoch 40	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3454	  cost 0.1358
Epoch 41	  lr 0.000050	  <font color="#EF2929"><b>EER</b></font> 1.3560	  cost 0.1447
Epoch 42	  lr 0.000005	  <font color="#EF2929"><b>EER</b></font> 1.2337	  cost 0.1385
Epoch 43	  lr 0.000005	  <font color="#EF2929"><b>EER</b></font> 1.2071	  cost 0.1338</pre>

## Step 3. Training Far-field model (finetuning with Vox2dev and HI-MIA data)

Modify the parameters in `./config/config_ecapatdnn_dist_ftmix.py` before training. Modify the training dir as `ftmix` and valuation dir as `HIMIA_eval`.

change the "start_epoch" as 44 (ecapatdnn pre-trained model as shown above) and running：

```shell
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 train_dist_ftmix.py &
```

## Step 4. Valuation

Modify `./config/config_scoring.py` and running with corresponding model number

```shell
python test.py --epoch 58
```

Then, submit the score.txt to the leaderboard.

![img](./DURUII/res/submit.png)
