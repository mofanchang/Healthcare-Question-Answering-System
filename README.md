# Healthcare-Question-Answering-System
Healthcare Information Retrieval-Augmented Generation System

# 醫療問答 RAG 系統

基於 Qwen 模型的醫療領域檢索增強生成（RAG）系統，專為繁體中文醫療問答設計。
![Medical RAG Architecture](images/MedicalRag.png)

## 系統架構

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用戶查詢      │ -> │   向量檢索      │ -> │   重排序        │
│  (繁體中文)     │    │  (Embedding)    │    │  (Reranker)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      |
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   回答生成      │ <- │   上下文整合    │ <- │   文檔過濾      │
│   (Qwen LLM)    │    │  (Context)      │    │  (Top-K)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘


用戶查詢 -> 向量檢索(Top-20) -> 重排序(Top-3) -> 上下文整合 -> 回答生成
   ↓            ↓                 ↓              ↓            ↓
繁體中文    Embedding檢索      Reranker       Context     Qwen LLM
```

## 技術特色

### 針對醫療領域優化
- **文檔增強**：同義詞擴展、術語標準化
- **繁簡轉換**：自動處理繁簡體醫療文獻
- **專業術語**：醫療領域關鍵詞同義詞映射
- **分塊策略**：基於句號的智能文檔分割

### 記憶體優化
- **量化推理**：4-bit 量化降低記憶體佔用
- **批次處理**：分批處理避免記憶體溢出
- **顯存管理**：主動清理 CUDA 快取
- **模型卸載**：動態載入卸載機制

### 靈活架構
- **多重排序**： Reranker + 相似度重排序
- **可配置化**：統一配置管理
- **模組化設計**：可獨立替換各組件
- **錯誤處理**：完整的異常捕獲機制

## 模型組件

| 組件 | 模型 | 用途 |
|------|------|------|
| 主生成模型 | Qwen3-8B | 回答生成 |
| 嵌入模型 | Qwen3-Embedding-0.6B | 文檔向量化 |
| 重排序模型 | Qwen3-Reranker-0.6B | 相關性排序 |
| 向量資料庫 | ChromaDB | 向量儲存檢索 |

## 安裝部署

### 環境需求
```bash
# Python 3.8+
# CUDA 11.8+ (建議)
# 記憶體 16GB+ (建議 32GB+)
```

### 依賴安裝
```bash
pip install -r requirements.txt
```

### 快速啟動
```bash
python medical_rag.py
```

## 配置說明

### 基本配置
```python
@dataclass
class RAGConfig:
    llm_model_name: str = "Qwen/Qwen3-8B"           # 主模型
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"  # 嵌入模型
    reranker_model_name: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"  # 重排序模型
    
    # 檢索參數
    retrieval_top_k: int = 20        # 初始檢索數量
    rerank_top_k: int = 3            # 重排序後數量
    
    # 生成參數
    max_new_tokens: int = 150        # 最大生成長度
    generation_timeout: int = 40     # 生成超時時間
    
    # 記憶體優化
    embedding_batch_size: int = 16   # 嵌入批次大小
    chunk_size: int = 200            # 文檔分塊大小
    chunk_overlap: int = 50          # 分塊重疊長度
```

### 自訂醫療資料

替換 `example_docs` 中的範例資料：

```python
# 範例：載入您的醫療文件
def load_medical_documents(file_path: str) -> List[str]:
    """
    載入醫療文檔
    支援格式：txt, json, csv
    """
    documents = []
    # 實作您的文檔載入邏輯
    return documents

# 使用自訂資料
medical_docs = load_medical_documents("your_medical_data.txt")
processed_docs = doc_processor.process_documents(medical_docs)
vector_db.add_documents(processed_docs)
```

## 核心功能

### 1. 文檔處理與增強
```python
class DocumentProcessor:
    def chunk_document(self, text: str) -> List[str]:
        """智能文檔分塊，基於句號分割"""
        
    def augment_document(self, text: str) -> List[str]:
        """文檔增強，包含同義詞擴展"""
```

### 2. 向量檢索
```python
class VectorDatabase:
    def search(self, query: str, top_k: int) -> List[str]:
        """語義相似度檢索"""
        
    def add_documents(self, documents: List[Dict]):
        """批次添加文檔向量"""
```

### 3. 重排序機制
```python
class RerankerModel:
    def rerank(self, query: str, documents: List[str]) -> List[str]:
        """基於專業 Reranker 重排序"""

class SimpleSimilarityReranker:
    def rerank(self, query: str, documents: List[str]) -> List[str]:
        """基於餘弦相似度重排序"""
```

### 4. 回答生成與後處理
```python
class QwenMedicalRAG:
    def generate_answer(self, query: str, context: str) -> str:
        """生成醫療問答回答"""
        
    def clean_response(self, response: str) -> str:
        """清理和格式化回答"""
```

## 記憶體優化策略

### 1. 模型量化
```python
# 4-bit 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

### 2. 顯存管理
```python
# 主動清理顯存
torch.cuda.empty_cache()
gc.collect()
```

### 3. 批次處理
```python
# 分批處理嵌入
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    # 處理批次...
```

## 效能指標

### 硬體配置建議
| 配置級別 | GPU | 記憶體 | 預期效能 |
|----------|-----|--------|----------|
| 最低配置 | RTX 3060 12GB | 16GB | 基本可用 |
| 推薦配置 | RTX 4070 16GB | 32GB | 流暢運行 |
| 最佳配置 | A100 40GB | 64GB | 最佳效能 |

### 回答品質評估
- **準確性**：基於醫療知識庫的事實性回答
- **相關性**：通過重排序提升上下文相關性
- **一致性**：繁體中文醫療術語統一
- **安全性**：避免醫療建議的不當表述

## 擴展功能

### 1. 多語言支援
```python
# 新增其他語言轉換器
cc_en = OpenCC('s2t')  # 英文醫療術語
cc_jp = OpenCC('s2t')  # 日文醫療術語
```

### 2. 知識圖譜整合
```python
class MedicalKnowledgeGraph:
    def enhance_query(self, query: str) -> str:
        """基於醫療知識圖譜增強查詢"""
```

### 3. 多模態支援
```python
class MultiModalProcessor:
    def process_medical_images(self, images: List[str]) -> List[str]:
        """處理醫療影像資料"""
```

## 使用範例

### 基本問答
```python
# 初始化系統
rag_system = QwenMedicalRAG(config)

# 查詢示例
question = "便秘的治療方法有哪些？"
answer = rag_system.query(question)
print(answer)
```

### Web 介面
```python
# 啟動 Gradio 介面
iface = gr.Interface(
    fn=rag_system.query,
    inputs="text",
    outputs="text",
    title="醫療問答RAG系統"
)
iface.launch()
```

## 故障排除

### 常見問題

**Q: 模型載入失敗**
```bash
# 檢查 CUDA 版本
nvidia-smi
# 檢查 PyTorch 版本
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: 記憶體不足**
```python
# 調整批次大小
config.embedding_batch_size = 8  # 降低批次大小
config.max_length = 256          # 減少序列長度
```

**Q: 生成回答品質差**
```python
# 調整檢索參數
config.retrieval_top_k = 30      # 增加檢索範圍
config.rerank_top_k = 5          # 增加重排序數量
```

## 開發路線圖

- [ ] 多輪對話支援
- [ ] 醫療影像整合
- [ ] 知識圖譜增強
- [ ] 多語言擴展
- [ ] 移動端部署
- [ ] 雲端服務整合



## 授權聲明

本專案僅供技術展示和學術研究使用，不提供醫療建議。實際醫療問題請諮詢專業醫療人員。




