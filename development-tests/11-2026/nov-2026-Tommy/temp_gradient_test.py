import numpy as np

def normalize(vec):
    if np.linalg.norm(vec) == 0:
        return np.array([0, 0])
    return vec / np.linalg.norm(vec)

def cosine_similarity(a, b):
    if np.linalg.norm(a) * np.linalg.norm(b) < 10e-6:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def project(vec_1, vec_2):
    if np.linalg.norm(vec_2) < 10e-6:
        return np.array([0, 0])
    return np.dot(vec_2, vec_1) / np.dot(vec_2, vec_2) * vec_2

def correction_1(vectors, index):
    current = vectors[index]
    sum_conflicting = np.array([0.0, 0.0])
    for i in range(len(vectors)):
        if i == index: continue
        other = vectors[i]
        if cosine_similarity(current, other) < 0:
            sum_conflicting += other
    # print(sum_conflicting)
    projected_vector = project(current, sum_conflicting)
    # print(f"Projected vector: {projected_vector}")
    return current - projected_vector

def correction_2(vectors, index):
    current = vectors[index]
    sum_conflicting = np.array([0.0, 0.0])
    for i in range(len(vectors)):
        if i == index: continue
        other = vectors[i]
        similarity = cosine_similarity(current, other)
        if similarity < 0:
            sum_conflicting += other * similarity
    # print(sum_conflicting)
    # projected_vector = project(current, sum_conflicting)
    # print(f"Projected vector: {projected_vector}")
    return current - sum_conflicting

def correction_3(vectors, index):
    current = vectors[index]
    sum_nonconflicting = np.array([0.0, 0.0])
    sum_conflicting = np.array([0.0, 0.0])
    for i in range(len(vectors)):
        other = vectors[i]
        similarity = cosine_similarity(current, other)
        if similarity < 0:
            sum_conflicting += other * similarity
        else:
            sum_nonconflicting += other * similarity
    # print(sum_conflicting)
    # projected_vector = project(current, sum_conflicting)
    # print(f"Projected vector: {projected_vector}")
    return sum_nonconflicting - sum_conflicting

def correction_4(vectors, index):
    current = vectors[index]
    sum_nonconflicting = np.array([0.0, 0.0])
    sum_conflicting = np.array([0.0, 0.0])
    for i in range(len(vectors)):
        other = vectors[i]
        similarity = cosine_similarity(current, other)
        if similarity < 0:
            sum_conflicting += other * similarity
        else:
            sum_nonconflicting += other * similarity
    # print(sum_conflicting)
    # projected_vector = project(current, sum_conflicting)
    # print(f"Projected vector: {projected_vector}")
    return normalize(sum_nonconflicting) - normalize(sum_conflicting)

def correction_5(vectors, index):
    current = vectors[index]
    sum_conflicting = np.array([0.0, 0.0])
    for i in range(len(vectors)):
        if i == index: continue
        other = vectors[i]
        similarity = cosine_similarity(current, other)
        if similarity < 0:
            sum_conflicting += other * similarity
    # print(sum_conflicting)
    projected_vector = project(current, sum_conflicting)
    # print(f"Projected vector: {projected_vector}")
    return current - projected_vector

vec_a = np.array([1.0, 0.0])

vec_b_options = np.array([
    [np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))],
    [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
    [np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120))],
    [np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135))],
    [np.cos(np.deg2rad(180)), np.sin(np.deg2rad(180))],
])

vec_c_options = np.array([
    [np.cos(np.deg2rad(-60)), np.sin(np.deg2rad(-60))],
    [np.cos(np.deg2rad(-90)), np.sin(np.deg2rad(-90))],
    [np.cos(np.deg2rad(-120)), np.sin(np.deg2rad(-120))],
    [np.cos(np.deg2rad(-135)), np.sin(np.deg2rad(-135))],
    [np.cos(np.deg2rad(-180)), np.sin(np.deg2rad(-180))],
])

for vec_b in vec_b_options:
    for vec_c in vec_c_options:
        vectors = [vec_a, vec_b, vec_c]
        print('-'*120)
        print(vectors)
        original_sum = vec_a + vec_b + vec_c
        for i in range(len(vectors)):
            print(f"Similarity of original gradient to class {i}: {round(cosine_similarity(original_sum, vectors[i]), 5)}")

        corrected_vectors = []
        for i in range(len(vectors)):
            ##################################################
            corrected_vectors.append(correction_1(vectors, i))
            ##################################################
        corrected_sum = np.sum(corrected_vectors, axis=0)
        print(corrected_sum)
        for i in range(len(vectors)):
            print(f"Similarity of corrected gradient to class {i}: {round(cosine_similarity(corrected_sum, vectors[i]), 5)}")

