import numpy as np
import json
import requests

import sys


if __name__=="__main__":
    while True:
        q1 = input("q1")
        q2 = input("q2")
        result = requests.get("http://localhost:5000/docvec",params={'q':f"{q1}\n{q2}"})
        data = result.json()
        try:
            assert len(data)==2
            assert len(data[0])==1024
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()

        data = np.array(data,dtype='float')

        def get_similarity(a1,a2):
            a1 = np.array(a1)
            a2 = np.array(a2)
            return np.sum(a1*a2)/(np.linalg.norm(a1)*np.linalg.norm(a2))

        def euclidean_distance_similarity(a1,a2):
            a1 = np.array(a1)
            a2 = np.array(a2)
            return np.linalg.norm(a2-a1)/(np.linalg.norm((a1+a2)/2))
        print(f"Cosine Similarity: {get_similarity(data[0],data[1])}")
        print(f"Euclidean Similarity: {1-euclidean_distance_similarity(data[0],data[1])}")