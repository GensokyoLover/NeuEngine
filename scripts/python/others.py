import numpy as np
import json
# 1. 输入三维向量列表
with open(r"H:\Falcor\scripts\python\scenes\onlybunny\level3/position.json","r") as f:
    vectors = json.load(f)


# 转成 numpy 数组
vectors = np.array(vectors)

# 2. normalize 成单位向量
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
unit_vectors = vectors / norms

# 3. 给定一个目标向量
query = np.array([0.8432, 0.2931, 1.2314])
query = query / np.linalg.norm(query)  # normalize 查询向量

# 4. 计算与每个单位向量的相似度（点积越大越接近）
similarities = unit_vectors @ query  # 等价于 cos(theta)

# 5. 找到最相似的三个向量的下标
top3_idx = np.argsort(-similarities)[:3]  # 负号表示从大到小排序

print("最接近的三个下标:", top3_idx)
print("对应相似度:", similarities[top3_idx])