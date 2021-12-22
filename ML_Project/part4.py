import sys
from math import log

min_num = -(sys.float_info.max)
def take_log(x):
    if x == 0:
        return min_num
    else:
        return log(x)

def train_emit_part4(data):
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
            if tmp==0:
                e_prob[i][j] = 0
            else:
                e_prob[i][j] = tmp / (e_count[i])
        
        # handle unseen words based on distribution of states for known words
        known_prob = e_count[i] / sum(e_count.values())
        e_prob[i]["#UNK#"] = known_prob
    
    return e_prob
    
def initialize_train_part4(val):
    state_ls = ['START', 'O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative', 'STOP']

    q_dict = {}
    for state1 in state_ls:
        for state2 in state_ls:
            if state2!="START" and state1!="STOP":
                q_dict[(state1, state2)] = val
            
    q_dict[("PRESTART", "START")] = val
    return q_dict


def train_part4(data):
    
    q_prob = initialize_train_part4({})
    q_count = initialize_train_part4(0)

    prev_prev_tag = "PRESTART"
    prev_tag = "START"
    for line in data:
        content = line.strip().rsplit(' ', 1)
            
        # empty line/char indicates end of sentence
        if len(content)==1:
            if q_prob[(prev_prev_tag, prev_tag)].get("STOP"):
                q_prob[(prev_prev_tag, prev_tag)]["STOP"] += 1
            else:
                q_prob[(prev_prev_tag, prev_tag)]["STOP"] = 1
            q_count[(prev_prev_tag, prev_tag)] += 1
            prev_prev_tag = "PRESTART"
            prev_tag = "START"
            continue
            
        curr_tag = content[-1]
        if q_prob[(prev_prev_tag, prev_tag)].get(curr_tag):
            q_prob[(prev_prev_tag, prev_tag)][curr_tag] += 1
        else:
            q_prob[(prev_prev_tag, prev_tag)][curr_tag] = 1
        q_count[(prev_prev_tag, prev_tag)] += 1
        
        prev_prev_tag = prev_tag
        prev_tag = curr_tag

    for i in q_prob.keys():
        for j in q_prob[i].keys():
            if q_count[i]>0:
                q_prob[i][j] /= q_count[i]
        
    return q_prob

def pred_part4(sentence, e_prob, q_prob):
    
    pie = {}
    pie[0] = initialize_train_part4(min_num)
    pie[0][("PRESTART", "START")] = 1
    path = {0:("PRESTART", "START")}

    i = 1
    max_path = min_num
    
    while i<len(sentence)+1:
        pie[i] = {}
        path[i] = {}
        # current state
        for u in ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']:
            max_path = 'O'
            # previous state
            for v in ['START', 'O','B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']:
                # 2nd previous state
                for t in ['PRESTART', 'START', 'O','B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']:
                    q = 0 if q_prob.get((t, v), -1) == -1 else q_prob[(t, v)].get(u,0)
                    p = pie[i-1].get((t, v), min_num) + take_log(q * e_prob[u].get(sentence[i-1],e_prob[u]["#UNK#"]))

                    if p > pie[i].get((v, u),min_num):
                        max_path = t  
                    
                    q = 0 if q_prob.get((max_path, v), -1) == -1 else q_prob[(max_path, v)].get(u,0)
                    pie[i][(v, u)] = pie[i-1].get((max_path, v), min_num) + take_log(q * e_prob[u].get(sentence[i-1],e_prob[u]["#UNK#"]))
                    path[i][(v, u)] = max_path
    
        i+=1
        
    
    u = "STOP"
    max_path = 'O'
    pie[i] = {}
    path[i] = {}
    for v in ['O','B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']:
        for t in ['O','B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']:
            q = 0 if q_prob.get((t,v), -1) == -1 else q_prob[(t, v)].get(u,0)
            p = pie[i-1].get((t, v), min_num) + take_log(q)
            
            if p > pie[i].get((v, u),min_num):
                max_path = t   
            
            q = 0 if q_prob.get((max_path, v), -1) == -1 else q_prob[(max_path, v)].get(u,0)
            pie[i][(v, u)] = pie[i-1].get((max_path, v), min_num) + take_log(q)
            path[i][(v, u)] = max_path
    
    best_path = []
    state_2seq = max(pie[i], key=pie[i].get)
    while i!=0:
        best_path.append(state_2seq[0])
        state_2seq = (path[i][state_2seq], state_2seq[0])
        i-=1
    best_path.reverse()
        
    return best_path[1:]


def main():
    print("Training on:\t",train_file)
    print("Predicting for:\t",test_file)
    print(f"Running for {dataset}...")
           
    f = open(train_file, "r", encoding="utf8")
    data = []
    for r in f:
        data.append(r.strip("\n"))

    e_prob = train_emit_part4(data)
    q_prob = train_part4(data)
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
        pred += pred_part4(i, e_prob, q_prob)
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
        print("Usage: python part4.py <dataset> <optional: test_file>")
        
    dataset = sys.argv[1]
    assert(dataset in ["ES", "RU"])
        
    train_file = f"inputData/{dataset}/train"
    test_file = f"inputData/{dataset}/dev.in"
    out_type = test_file.split("/")[-1].split(".")[0]
    out_file = f"{dataset}/{out_type}.p4.out"
    
    if len(sys.argv)==3:
        test_file = sys.argv[2]
        out_type = test_file.split("/")[-1].split(".")[0]
        out_file = f"{dataset}/{out_type}.p4.out"
        
    main()