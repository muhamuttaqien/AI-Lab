Transformer-Based Models: Transformer-XL, GPT, BERT, GPT-2, XLNet

# ---------------
# Transformer-XL
# ---------------
- Larger input length to avoid context fragmentation issue on input sequence

# -----
# GPT 1
# -----
- Trained in semi-supervised with datasets from books and wikipedia for only language modelling task
- Pre-trained model capable of empowering many models later trained in supervised way (spam classifier)
- Decoder-only block/ transformer
- Normal self-attention
- Auto-regression capability
- No bidirectional capability
- 12 decoder without the encoder-decoder attention

# -----
# BERT
# -----
- Trained in semi-supervised with datasets from books and wikipedia for language modelling task
- Trained also for two-sentence classification task to better recognize relationships between multiple sentence
- Pre-trained model capable of empowering many models later trained in supervised way (spam classifier)
- Also mentioned as masked-language-model
- Very good at fill-in-the-blanks
- We can also use the pre-trained BERT to create contextualized word embeddings
- Encoder-only block/ transformer
- Normal self-attention + <MASK> mechanism masking 15% of words in the input and try to predict
- Bidirectional capability
- VANILLA Transformer: 6 encoder, 512 ffnn, 8 heads
- BERT-BASE Transformer: 12 encoder, 768 ffnn, 12 heads
- BERT-LARGE TRansformer: 24 encoder, 1028 ffnn, 16 heads

# -----
# GPT 2
# -----
- GPT 1
- Very good at writing essays
- Masked self-attention + Mask Interfering mechanism
- 1.5 billion parameters Trained by WebText 40GB OpenAI researchers crawled from the internet as part of the research effort
- Also trained using + 100 GPU
- GPT-2 SMALL consists of 117M parameters while GPT-2 EXTRA LARGE consists of 1,5B parameters