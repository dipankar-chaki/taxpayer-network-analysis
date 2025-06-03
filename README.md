![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

# 🕸️ Graph-Based Taxpayer Network Analysis

This project explores **graph analytics and role detection** techniques to identify central actors, potential shell entities, and anomalous behavior in simulated taxpayer transaction data. It mimics how tax authorities (like the ATO) might uncover hidden financial patterns using network science and unsupervised learning.

---

## 🔍 Project Goals

- Model transactions between entities as a **directed graph**
- Detect **influential actors**, **anomalous nodes**, and **hidden communities**
- Apply **role detection** using **Node2Vec embeddings**
- Compare classic **centrality-based** and **ML-based** anomaly detection methods
- Lay the foundation for **Graph Neural Network (GNN)** extensions

---

## 🧱 Project Structure
```text
graph-taxpayer-network/
├── data/
│   ├── raw/                # Original PaySim dataset
│   ├── processed/          # Cleaned subset, saved graphs, embeddings
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_graph_building.ipynb
│   ├── 04_node2vec_roles.ipynb
│   ├── 05_anomaly_comparison.ipynb
├── scripts/
│   └── utils.py            # Helper functions for graphs, embeddings, clustering
├── requirements.txt
└── README.md
```
---

## 📦 Setup Instructions

### ✅ 1. Clone & Set Up Environment
```bash
git clone https://github.com/dipankar-chaki/graph-taxpayer-network.git
cd graph-taxpayer-network
```

### Create a virtual environment
```bash
python3 -m venv graph-env
source graph-env/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### ✅ 2. Download Dataset
```bash
pip install kaggle  # if not installed
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download and unzip the dataset
kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip
```

### ✅ 3. Project Steps
📁 Step 1: Data Inspection
- Explore and clean PaySim transaction logs
- Focus on sender → receiver behavior
- Save a smaller subset (5,000 rows) for rapid prototyping

🧹 Step 2: Preprocessing
- Extract sender/receiver and amount
- Filter out self-transfers and zero-value transactions
- Create edge list for graph construction

🕸️ Step 3: Graph Construction & Centrality Analysis
- Build a directed graph using NetworkX
- Compute centrality metrics:
- Degree Centrality
- Betweenness Centrality
- Eigenvector Centrality
- Visualize central actors

🧠 Step 4: Role Detection using Node2Vec
- Learn low-dimensional node embeddings
- Apply KMeans clustering on embeddings to discover roles
- Identify role-based outliers (anomalies far from cluster centers)

🧮 Step 5: Compare Anomalies

Merge anomalies from:
- Centrality metrics
- Node2Vec outliers

Visualize:
- Venn diagrams of overlap
- t-SNE plot of node embeddings with anomaly highlights
- Bar plot of anomaly types: Centrality, Role, Both

---

## 🙈 .gitignore
This project includes a .gitignore file that excludes:
Virtual environments (venv/)
Bytecode files (__pycache__/)
Notebook checkpoints
System files like .DS_Store
Large output files (e.g., .csv exports)

---

## 📄 License
MIT — use freely with attribution.

---

## 🙋‍♂️ Author

**Dipankar Chaki**  
PhD in Computer Science | ML & AI Researcher
