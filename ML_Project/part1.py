import sys

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

def pred_part1(word, e_prob):
    
    ans = {}
    for i in e_prob.keys():
        ans[e_prob[i].get(word,e_prob[i]["#UNK#"])] = i
    return ans[max(ans.keys())]

def main():
    print("Training on:\t",train_file)
    print("Predicting for:\t",test_file)
    print(f"Running for {dataset}...")
           
    f = open(train_file, "r", encoding="utf8")
    data = []
    for r in f:
        data.append(r.strip("\n"))

    e_prob = train_part1(data)
    f.close()

    f = open(test_file, "r", encoding="utf8")
    data = []
    lines = []
    for r in f:
        lines.append(r)
        data.append(r.strip("\n"))
    data = [x for x in data if x]
    pred = []
    for i in data:
        pred.append(pred_part1(i, e_prob))
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
        print("Usage: python part1.py <dataset> <optional: test_file>")
        
    dataset = sys.argv[1]
    assert(dataset in ["ES", "RU"])
        
    train_file = f"inputData/{dataset}/train"
    test_file = f"inputData/{dataset}/dev.in"
    out_type = test_file.split("/")[-1].split(".")[0]
    out_file = f"{dataset}/{out_type}.p1.out"
    
    if len(sys.argv)==3:
        test_file = sys.argv[2]
        out_type = test_file.split("/")[-1].split(".")[0]
        out_file = f"{dataset}/{out_type}.p1.out"
        
    main()
    