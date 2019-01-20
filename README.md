# Shopping

모델을 훈련하는 코드와 설명을 포함합니다.
훈련된 모델은 GCS (Google Cloud Storage) 를 통해 다운로드 받을 수 있습니다.

전체 과정은 크게 SentencePiece 를 훈련하고, TFRecord 파일을 생성하고, TensorFlow 모델을 훈련하고, 예측하는 부분으로 나뉘어집니다.

### 설치

먼저 Python 3.6 이상이 설치되어있는지 확인하고 다음과 같이 필요한 패키지를 설치해주세요.

```
pip install -r requirements.txt
```

### TSV 로 변환

`text_dump.py` 는 H5 형식의 데이터를 TSV 형식으로 변환하는 스크립트입니다.
사용 방법은 다음과 같습니다.

```
python text_dump.py \
  --input_file=train.chunk.01 \
  --output_file=train.chunk.01.tsv
```

대회에서 제공된 모든 train, dev, test 데이터파일을 이렇게 TSV 로 변환합니다.

### Text 정규화

먼저 다음과 같이 trian, dev 의 text feature 를 추출합니다.

```
cat train.chunk.0[1-9].tsv dev.chunk.01.tsv \
  | awk -F$'\t' -v OFS=$'\n' '{ print $2, $3, $4, $5 }' \
  | awk NF \
  | uniq > features.txt
```

그리고 다음과 같이 추출된 text feature 를 정규화하고 무작위로 섞어줍니다.

```
python normalize.py --input_file=features.txt --output_file=normalized.txt
shuf normalized.txt > normalized_shuffled.txt
```

### SentencePiece 훈련

[SentencePiece](https://github.com/google/sentencepiece) 는 문장을 토큰으로 나누어주는 tokenizer 입니다.
SentencePiece 를 설치한 후 다음과 같이 tokenizer 를 훈련합니다.

```
spm_train --input=normalized_shuffled.txt \
  --model_prefix=sentpiece \
  --vocab_size=100000 \
  --character_coverage=0.9995
```

### TFRecord 데이터 생성

훈련 중 모델 evaluation 에 사용하기 위해 훈련 PID 100,000 개를 무작위로 선택합니다.

```
cat train.chunk.0[1-9].tsv \
  | cut -d$'\t' -f1 \
  | shuf -n100000 > id.dev.txt
```


TensorFlow 모델 훈련은 [T2T](https://github.com/tensorflow/tensor2tensor) 를 이용합니다.
먼저 다음과 같이 T2T 에서 사용할 훈련 데이터를 생성합니다.

```
mkdir -p ~/t2t_tmp/shopping
mkdir -p ~/t2t_data/shopping
mv *.chunk.0[0-9] id.dev.txt ~/t2t_tmp/shopping/
wget https://storage.googleapis.com/kakao-arena/t2t_data/hierarchical_shopping_private_lb/labels.json
cp labels.json sentpiece.model ~/t2t_data/shopping/
cp cate1.json ~/t2t_data/shopping/category.json

t2t-datagen \
  --t2t_usr_dir=~/kakao-arena-shopping/shopping \
  --tmp_dir=~/t2t_tmp/shopping \
  --data_dir=~/t2t_data/shopping \
  --problem=hierarchical_shopping
```

* `t2t_usr_dir`: 이 프로젝트의 `shopping` 모듈의 경로
* `tmp_dir`: h5 데이터 파일과 `id.dev.txt` 이 있는 디렉토리
* `data_dir`: TFRecord 파일을 저장할 디렉토리입니다. 아래 파일들을 미리 복사해두어야 합니다.
  * `category.json`: `cate1.json` 을 그대로 복사
  * `labels.json`: Array of labels (`[[1, 1, -1, -1], ...]`)
  * `sentpiece.model`: SentencePiece 모델

그리고 다음과 같이 public/private leaderboard 를 위한 데이터도 생성합니다.

```
mkdir -p ~/t2t_data/hierarchical_shopping_public_lb
cp labels.json sentpiece.model ~/t2t_data/hierarchical_shopping_public_lb/
cp cate1.json ~/t2t_data/hierarchical_shopping_public_lb/category.json

t2t-datagen \
  --t2t_usr_dir=~/kakao-arena-shopping/shopping \
  --tmp_dir=~/t2t_tmp/shopping \
  --data_dir=~/t2t_data/hierarchical_shopping_public_lb \
  --problem=hierarchical_shopping_public_lb
```

```
mkdir -p ~/t2t_data/hierarchical_shopping_private_lb
cp labels.json sentpiece.model ~/t2t_data/hierarchical_shopping_private_lb/
cp cate1.json ~/t2t_data/hierarchical_shopping_private_lb/category.json

t2t-datagen \
  --t2t_usr_dir=~/kakao-arena-shopping/shopping \
  --tmp_dir=~/t2t_tmp/shopping \
  --data_dir=~/t2t_data/hierarchical_shopping_private_lb \
  --problem=hierarchical_shopping_private_lb
```

### 모델 훈련

다음과 같이 T2T 를 이용해서 TensorFlow 모델을 훈련합니다.

```
t2t-trainer \
  --t2t_usr_dir=~/kakao-arena-shopping/shopping \
  --data_dir=~/t2t_data/shopping \
  --output_dir=~/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu \
  --problem=hierarchical_shopping \
  --model=lstm_seq2seq_attention_bidirectional_encoder \
  --hparams_set=lstm_base_batch_8k_hidden_1k \
  --train_steps=200000 \
  --worker_gpu=8 \
  --schedule=train
```

### 모델 Averaging

최종 모델을 얻기 위해 5 개 checkpoints 의 평균을 구합니다.

```
python avg_checkpoints.py \
  --checkpoints=19000,20000,21000,22000,23000 \
  --prefix=~/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/model.ckpt- \
  --output_path=~/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt
```

### 예측 모델 생성

불필요한 Adam momentum 과 variance 를 제거한 예측용 checkpoint 를 생성합니다.

```
python reduce_checkpoint_size.py \
  --input_path=~/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0 \
  --output_path=~/t2t_train_reduced/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0
```


### 예측

`predict.py` 는 훈련된 모델을 이용해서 예측하고 제출용 TSV 를 만드는 스크립트입니다.
다음과 같이 public leaderboard 를 위한 TSV 파일을 만듭니다.

```
python predict.py \
  --data_dir=~/t2t_data/hierarchical_shopping_public_lb \
  --problem=hierarchical_shopping_public_lb \
  --model=lstm_seq2seq_attention_bidirectional_encoder \
  --hparams_set=lstm_base_batch_8k_hidden_1k \
  --checkpoint_path=~/t2t_train_reduced/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0 \
  --output_file=hierarchical.lstm.base_batch_8k_hidden_1k_8gpu.19-23k.public.tsv
```

다음과 같이 private leaderboard 에 제출된 결과를 재현할 수 있습니다.

```
python predict.py \
  --data_dir=gs://kakao-arena/t2t_data/hierarchical_shopping_private_lb \
  --problem=hierarchical_shopping_private_lb \
  --model=lstm_seq2seq_attention_bidirectional_encoder \
  --hparams_set=lstm_base_batch_8k_hidden_1k \
  --checkpoint_path=gs://kakao-arena/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0 \
  --output_file=hierarchical.lstm.base_batch_8k_hidden_1k_8gpu.19-23k.private.tsv
```

훈련된 모델과 중간 파일들은 GCS 에 있습니다.
만약 GCS 를 사용하지 않을 경우 필요한 파일들을 내려받은 후 경로를 재지정해주어야 합니다.

* `data_dir`: TFRecord 파일이 있는 디렉토리입니다. 아래 파일들을 복사해야두어야 합니다.
  * [category.json](https://storage.googleapis.com/kakao-arena/t2t_data/hierarchical_shopping_private_lb/category.json)
  * [labels.json](https://storage.googleapis.com/kakao-arena/t2t_data/hierarchical_shopping_private_lb/labels.json)
  * [sentpiece.model](https://storage.googleapis.com/kakao-arena/t2t_data/hierarchical_shopping_private_lb/sentpiece.model)
  * [hierarchical_shopping_private_lb-test-00000-of-00002](https://storage.googleapis.com/kakao-arena/t2t_data/hierarchical_shopping_private_lb/hierarchical_shopping_private_lb-test-00000-of-00002)
  * [hierarchical_shopping_private_lb-test-00001-of-00002](https://storage.googleapis.com/kakao-arena/t2t_data/hierarchical_shopping_private_lb/hierarchical_shopping_private_lb-test-00001-of-00002)
* `checkpoint_path`: 훈련된 모델의 checkpoint 경로입니다. 아래 파일들을 임의의 디렉토리에 복사하고, `checkpoint_path` 를 해당 디렉토리명 + `/averaged.19-23k.ckpt-0` 로 지정해야 합니다.
  * [averaged.19-23k.ckpt-0.meta](https://storage.googleapis.com/kakao-arena/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0.meta)
  * [averaged.19-23k.ckpt-0.index](https://storage.googleapis.com/kakao-arena/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0.index)
  * [averaged.19-23k.ckpt-0.data-00000-of-00001](https://storage.googleapis.com/kakao-arena/t2t_train/hierarchical_shopping/lstm/base_batch_8k_hidden_1k_8gpu/averaged.19-23k.ckpt-0.data-00000-of-00001)
* `output_file`: 생성할 TSV 파일 경로

나머지 인자들은 변경할 필요가 없습니다.

* `problem`: 데이터셋 이름
* `model`: 모델 이름
* `hparams_set`: Hyperparameter
