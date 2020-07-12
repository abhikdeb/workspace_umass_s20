import requests
# import json
import re
import os
import pandas as pd
import pickle
# import nltk

file_per250 = "D:/UMass/Spring20/685/Homework/data/per250.txt"
pck_data = "D:/UMass/Spring20/685/Homework/data/save.pickle"


def fetch_data():
    url = "http://hobbes.cs.umass.edu/~brenocon/anlp2020/hw1/Personal250/for_abhikdeb.txt"
    r = requests.get(url,allow_redirects=True)
    try:
        open(file_per250,'wb').write(r.content)
    except IOError:
        print("IOError")
    finally:
        print("Fetch Complete!")
    return True


def main():
    reset = False
    print("Running Main...")
    if not os.path.exists(file_per250):
        fetch_data()
    if (not os.path.exists(pck_data)) or reset:
        f_250 = open(file_per250, encoding='utf-8', mode='r', errors='replace')
        count = 0
        cols = ['Data', 'Length', 'ReTweet', 'Handle', 'Links', 'Type', 'Label', 'Notes']
        df = pd.DataFrame(columns=cols)
        while True:
            count += 1
            line = f_250.readline()
            if not line:
                break
            re_tweet = False
            links = None
            source = None
            if line[0:3] == "RT ":
                re_tweet = True
            if line.startswith("RT ") or line.startswith("@"):
                source = re.findall('@[^ :]*', line)[0]
            links = re.findall(r"http[s]?:\/\/[^ \n]*", line)
            if len(links) == 0:
                links = None
            df.loc[count-1] = [line.strip(), len(line.strip()), re_tweet, source, links, None, None, None]
        f_250.close()
        pickle.dump(df, open(pck_data, 'wb'))
    df = pickle.load(open(pck_data, 'rb'))
    print(df.shape)
    print(df.head)
    # Load Final data Labels
    dat = pd.read_excel('D:/UMass/Spring20/685/Homework/data/to_annotate.xlsx', sheet_name='Final_data')
    df['Label'] = dat['Label'].values
    print(df.head)
    pickle.dump(df, open(pck_data, 'wb'))
    return True


if __name__ == "__main__":
    main()

