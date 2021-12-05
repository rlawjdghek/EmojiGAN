# EmojiGAN for Final Project of Deep Generative Models

### Environment
pytorch==1.10.0\
torchvision==0.9.0\
transformers==4.12.5\
huggingface-hub==0.2.0\
einops=0.3.2\
tokenizers=0.10.3\
CUDA==11.3

### Dataset
Download images and a pickle file from [link](https://drive.google.com/drive/folders/1UD6cCWa6lxz9zJmiuaz7dKylMdLMVzl3?usp=sharing).
We manually select images related to faces. Also, we remove some enterprises which have very tiny images (4x4). 

### Train
```bash
python main.py --mode train --data_names <choose entriprises in 'Apple', 'Facebook', 'Google', 'JoyPixels', 'Samsung', 'Twitter', 'Windows'>
```

### Test from pickle 
```bash
python main.py --mode test_from_tokenizer --test_pickle_path <pickle file path for test>
```

### Test from tokenizer
You can generate emojitons from the Korean sentences. 
```bash
python main.py --mode test_from_tokenizer 
```


### Test with Pruned weight
You can use model pruning with pruned_rest_ratio.
```bash
python main.py --mode test_from_tokenizer --use_pruning --pruning_rest_ratio 0.7
```

