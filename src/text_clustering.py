import logging
import faiss
import sys
import numpy as np
from collections import defaultdict
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class TextClustering:
    def __init__(self, 
                 embed_model_name="all-MiniLM-L6-v2", 
                 embed_device="cpu", 
                 embed_batch_size=8, 
                 umap_components=2, 
                 umap_metric="cosine", 
                 dbscan_eps=0.8, 
                 dbscan_min_samples=20,
                 summary_chunk_size=480) -> None:
        
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_model = SentenceTransformer(self.embed_model_name, device=self.embed_device)
        self.embeddings = None
        self.embed_batch_size = embed_batch_size

        self.faiss_index = None
        
        self.umap_components = umap_components
        self.umap_metric = umap_metric

        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        self.summary_chunk_size = summary_chunk_size
    
    def fit(self, texts, embeddings=None):
        self.texts = texts
        if embeddings is None:
            logging.info("Embedding texts...")
            self.embeddings = self.embed(texts)
        else:
            logging.info("Using provided embeddings...")
            self.embeddings = embeddings
        
        logging.info("Building vectorstore index...")
        self.faiss_index = self.compute_faiss_index(self.embeddings)

        logging.info("Reducing dimensionality...")
        self.projections, self.umap = self.reduce(self.embeddings)

        logging.info("Clustering...")
        self.labels = self.cluster(self.projections)

        logging.info("Calculating cluster centers...")
        self.id_cluster_map = {index: label for index, label in enumerate(self.labels)}
        self.label_docs = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.label_docs[label].append(i)

        self.cluster_centers = self.get_cluster_centers()
        self.summaries = self.summarize(self.label_docs, 5)
        print(self.cluster_centers)


    def embed(self, texts):
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=self.embed_batch_size)
        return embeddings
    
    def compute_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def reduce(self, embeddings):
        mapper = UMAP(n_components=self.umap_components, metric=self.umap_metric).fit(embeddings)
        return mapper.embedding_, mapper
    
    def cluster(self, projections):
        cluster = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(projections)
        return cluster.labels_

    def get_cluster_centers(self):
        cluster_centers = {}
        for label in self.label_docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label_docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label_docs[label]])
            cluster_centers[label] = (x,y)
        return cluster_centers

    def summarize(self, label_docs, n_docs):
        summaries = {
            -1 : "Undefined"
        }
        unique_labels = len(set(label_docs)) - 1
        for label in range(unique_labels):
            top_n_texts = np.random.choice(label_docs[label], n_docs)
            for item in top_n_texts:
                print(self.texts[item][:self.summary_chunk_size])

    def save(self):
        pass



from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

""" loader = PyPDFLoader("./Slides.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

texts = []
for chunk, metadata, _ in splits:
    texts.append(chunk[1]) """

from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

TextClustering().fit(ds['text'][:1000])