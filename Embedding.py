from FlagEmbedding import BGEM3FlagModel

class EmbeddingService:
    def __init__(self):
        self.model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def encode(self,text, batch_size=12, max_length=8192):
        embeddings = self.model.encode(text, 
                                       batch_size=batch_size, 
                                       max_length=max_length)['dense_vecs']
        return embeddings