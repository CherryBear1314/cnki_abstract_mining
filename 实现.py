import jieba
import pandas as pd
import re
from collections import Counter
import networkx as nx
import pyvis.network as net
from gensim import corpora, models

#--------1 读取数据，分词预处理--------
df = pd.read_excel("CNKI-20260916150422373.xls")
# 加载自定义词典
domain_words = ["VR影像","具身认知","空间叙事","多模态交互","全景影像","眼动追踪","AIGC","元宇宙","沉浸感","交互叙事"]
for w in domain_words:
    jieba.add_word(w)
# 停用词
stopwords = set([line.strip() for line in open("stopwords.txt","r",encoding="utf‑8")])

def cut_doc(text):
    words = jieba.lcut(str(text))
    res = [w for w in words if len(w)>1 and w not in stopwords and not re.match(r"\d+",w)]
    return res

df["words"] = df["Summary‑摘要"].apply(cut_doc)
docs = df["words"].tolist()

# --------2 关键词共现网络--------
# 提取词频>=15的词
word_cnt = Counter([w for doc in docs for w in doc])
valid_words = {k for k,v in word_cnt.items() if v>=15}
# 构建共现矩阵
G = nx.Graph()
G.add_nodes_from(valid_words)
for doc in docs:
    doc_filter = list(set([w for w in doc if w in valid_words]))
    for i in range(len(doc_filter)):
        for j in range(i+1,len(doc_filter)):
            a,b = doc_filter[i],doc_filter[j]
            if G.has_edge(a,b):
                G[a][b]["weight"] +=1
            else:
                G.add_edge(a,b,weight=1)
# pyvis交互式网络图输出html
pv_net = net.Network(directed=False,height="700px",width="100%")
pv_net.from_nx(G)
pv_net.write_html("关键词共现网络.html")

# --------3 LDA主题聚类--------
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]
lda = models.LdaModel(corpus,id2word=dictionary,num_topics=5,random_state=88)
for idx,topic in lda.print_topics(num_words=10):
    print(f"主题{idx}: {topic}")

# --------4 逐年高频词统计--------
df["Year‑年"] = pd.to_numeric(df["Year‑年"],errors="coerce")
for yr in sorted(df["Year‑年"].dropna().unique()):
    sub = df[df["Year‑年"]==yr]["words"]
    allw = [w for doc in sub for w in doc]
    top = Counter(allw).most_common(15)
    print(f"\n{int(yr)}年Top15高频词：")
    print(top)
