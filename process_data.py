import xlrd
import re
import json
import random
random.seed(0)

def read_excel(path):
    #打开excel
    data = []
    wb = xlrd.open_workbook(path)
    #按工作簿定位工作表
    sh = wb.sheet_by_index(0)
    for _ in range(sh.nrows):
        data.append(sh.row_values(_))
    return data

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

if __name__=='__main__':
    data = read_excel('./data/raw.xlsx')[1:]
    rls = read_json('./data/rel_dict.json')
    save = [] 
    for _ in data:
        name1, name2, rl, content = _
        if len(name1)<len(name2):
            name1,name2 = _[1],_[0]
        content = re.sub(name1, "#%s#"%(name1), content, count=0)
        content = re.sub(name2, "$%s$"%(name2), content, count=0)
        keyindex =[]
        tag = rls[rl]

        for i,x in enumerate(content):
            if x=='#':
                keyindex.append(i)
            if len(keyindex)==2:
                break

        for i,x in enumerate(content):
            if x=='$':
                keyindex.append(i)
            if len(keyindex)==4:
                break

        if len(keyindex)!=4:
            continue
        save.append([keyindex, tag, content])

    random.shuffle(save)
    dev, train = save[:len(save)//5],save[len(save)//5:]

    print('%d %d'%(len(train), len(dev)))

    with open('./data/train_p.json','w',encoding='utf=8') as f:
        f.write(json.dumps(train, ensure_ascii=False))

    with open('./data/dev_p.json','w',encoding='utf=8') as f:
        f.write(json.dumps(dev, ensure_ascii=False))
    