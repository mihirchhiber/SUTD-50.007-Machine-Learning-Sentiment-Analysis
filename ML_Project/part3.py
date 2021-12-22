import sys
import numpy as np
import random
from math import log

min_num = -(sys.float_info.max)
def take_log(x):
    if x == 0:
        return min_num
    else:
        return log(x)
    
def train_part1(data, k=1):
    obs_set = set()
    e_prob = {'O':{}, 'B-positive':{}, 'B-neutral':{}, 'B-negative':{}, 'I-positive':{}, 'I-neutral':{}, 'I-negative':{}}
    e_count = {'O':0, 'B-positive':0, 'B-neutral':0, 'B-negative':0, 'I-positive':0, 'I-neutral':0, 'I-negative':0}

    for i in data:
        content = i.split(" ")
        if len(content)!=1:
            obs = " ".join(content[:-1])
            state = content[-1]
            if e_prob[state].get(obs, -1) != -1:
                e_prob[state][obs] += 1
            else:
                e_prob[state][obs] = 1
            e_count[state] += 1
            obs_set.add(" ".join(content[:-1]))
    

    for i in e_prob.keys():
        for j in obs_set:
            tmp = e_prob[i].get(j, 0) 
            e_prob[i][j] = tmp / (e_count[i] + k)
        e_prob[i]["#UNK#"] = k / (e_count[i] + k)
        
    return e_prob

def train_part2(data):
    q_prob = {'START':{} ,'O':{}, 'B-positive':{}, 'B-neutral':{}, 'B-negative':{}, 'I-positive':{}, 'I-neutral':{}, 'I-negative':{}}
    q_count = {'START': 0, 'O':0, 'B-positive':0, 'B-neutral':0, 'B-negative':0, 'I-positive':0, 'I-neutral':0, 'I-negative':0}

    data = ['temp START'] + data
    i = 0
    while i < len(data):
        if data[i] == "":
            data = data[:i] + ["temp STOP", 'temp START'] + data[i+1:]
        i+=1
    data = data[:-1]
    
    i = 0
    while i < len(data)-1:
        prev_tag = data[i].split(" ")[-1]
        curr_tag = data[i+1].split(" ")[-1]
        if prev_tag == "STOP":
            i+=1
            continue
        if q_prob[prev_tag].get(curr_tag):
            q_prob[prev_tag][curr_tag] += 1
        else:
            q_prob[prev_tag][curr_tag] = 1
        q_count[prev_tag] += 1
        i+=1

    for i in q_prob.keys():
        for j in q_prob[i].keys():
            q_prob[i][j] /= q_count[i]
        
    return q_prob

def pred_part3(sentence, e_prob, q_prob, top_k=1):
    random.seed(42)
    
    pie = {0:{'START': 1, 'O':0, 'B-positive':0, 'B-neutral':0, 'B-negative':0, 'I-positive':0, 'I-neutral':0, 'I-negative':0}}
    path = {0:['START','START']}
    state_ls = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']
    
    # initialize
    for state in pie[0].keys():
        pie[0][state] = [pie[0][state]]*top_k 
    pie[0]['START'] = [1] + [0]*(top_k-1)
    
    i = 1
    max_path = 0
    while i<len(sentence)+1:
        pie[i] = {}
        path[i] = {}
        # current state
        for u in state_ls:
            max_path = random.choice(state_ls)
            # previous state
            pie[i][u] = [-1]*top_k 
            path[i][u] = ['']*top_k 
            for v in ['O','B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative', 'START']:
                
                for r in range(top_k):
                    p = pie[i-1].get(v, [0]*top_k)[r] * q_prob[v].get(u,0) * e_prob[u].get(sentence[i-1],e_prob[u]["#UNK#"])
                    for a in range(top_k):
                        if p > pie[i].get(u,0)[a]:
                            if not ([v,r] in path[i][u]):
                                pie[i][u][a+1:] = pie[i][u][a:-1]
                                pie[i][u][a] = p
                                path[i][u][a+1:] = path[i][u][a:-1]
                                path[i][u][a] = [v,r]
                                break
        i+=1
        
    # terminate
    u = "STOP"
    pie[i] = {}
    path[i] = {}
    pie[i][u] = [-1]*top_k 
    path[i][u] = ['']*top_k 
    
    for v in ['O','B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative', 'START']:
        for r in range(top_k):
            p = pie[i-1].get(v, [0]*top_k)[r] * q_prob[v].get(u,0) 
            for a in range(top_k):
                if p > pie[i].get(u,0)[a]:
                    if not ([v,r] in path[i][u]):
                        pie[i][u][a+1:] = pie[i][u][a:-1]
                        pie[i][u][a] = p
                        path[i][u][a+1:] = path[i][u][a:-1]
                        path[i][u][a] = [v,r]
                        break
                    
    best_paths = []
    best_paths.append(path[i][u])
    u = path[i][u]
            
    i-=1
    while i!=0:
        temp = []
        for j in u:
            temp.append(path[i][j[0]][j[1]])
        best_paths.append(temp)
        u = temp
        i-=1
    
    for i in range(len(best_paths)):
        for j in range(len(best_paths[0])):
            best_paths[i][j] = best_paths[i][j][0]    
        
    best_paths = np.array(best_paths).T.tolist()
    best_paths = [i[::-1][1:] for i in best_paths]
    
    return best_paths[top_k-1]

def main():
    print("Training on:\t",train_file)
    print("Predicting for:\t",test_file)
    print(f"Running for {dataset}...")
    
    f = open(train_file, "r", encoding="utf8")
    data = []
    for r in f:
        data.append(r.strip("\n"))

    e_prob = train_part1(data)
    q_prob = train_part2(data)
    f.close()

    f = open(test_file, "r", encoding="utf8")
    data = []
    lines = []
    temp = []
    for r in f:
        lines.append(r)
        if r == "\n":
            data.append(temp)
            temp = []
            continue
        temp.append(r.strip("\n"))
    pred = []
    for i in data:
        pred += pred_part3(i, e_prob, q_prob, top_k=5)
    f.close()
           
    f = open(out_file,"w", encoding="utf8")
    j=0
    for i in range(len(lines)):
            word = lines[i].strip()
     
            if word:
                result = pred[j]
                f.write(word + " " + result)
                j+=1
            
            f.write("\n")
        
    f.close()
    print("Result at:\t",out_file)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python part3.py <dataset> <optional: test_file>")
        
    dataset = sys.argv[1]
    assert(dataset in ["ES", "RU"])
        
    train_file = f"inputData/{dataset}/train"
    test_file = f"inputData/{dataset}/dev.in"
    out_type = test_file.split("/")[-1].split(".")[0]
    out_file = f"{dataset}/{out_type}.p3.out"
    
    if len(sys.argv)==3:
        test_file = sys.argv[2]
        out_type = test_file.split("/")[-1].split(".")[0]
        out_file = f"{dataset}/{out_type}.p3.out"
        
    main()