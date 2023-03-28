from operator import index
import pandas as pd
import numpy as np
import spacy

#spacy==2.3.5

nlp = spacy.load("en_core_web_sm")
     
df = pd.read_excel("~/Downloads/Murali_coverage.xlsx",sheet_name="Sheet1")
qced_file = pd.read_csv("qced_data.csv",sep="\t")
qced_file = qced_file.dropna()

df_feature_opinion = pd.read_csv("feature_opinion_content_short.csv",sep="\t")
df_feature_opinion = df_feature_opinion.dropna()
df_feature_opinion = df_feature_opinion.drop_duplicates(subset=["feature_keywords","feature_category_insight_keywords","sentence_text"])
#df_feature_opinion.rename(columns = {'CONTENT':'sentence_text','FEATURE':'feature_keywords','OPINION':'feature_category_insight_keywords'}, inplace = True)

df["feature_category_insight_keywords"] = np.where(df["feature_category_insight_keywords"].isnull(), df["Ow"],df["feature_category_insight_keywords"])
df["feature_keywords"] = np.where(df["feature_keywords"].isnull(), df["Fw"],df["feature_keywords"])

mdu_aspect = pd.read_csv("mdu_aspects.csv")

df = df[["sentence_text", "feature_category_insight_keywords", "feature_keywords"]]
df = pd.concat([df,qced_file,df_feature_opinion, mdu_aspect])

def is_feature_opinion_exists(row):
    
    feature_word = row["feature_keywords"]
    opinion_word = row["feature_category_insight_keywords"]
    sentence = row["sentence_text"]

    if type(feature_word) != float  and feature_word in sentence and type(opinion_word) != float and opinion_word in sentence:
        return True

    return False

df["is_exists"] = df.apply(is_feature_opinion_exists,axis=1)
df = df[df["is_exists"]==True]

df = df[["sentence_text","feature_category_insight_keywords","feature_keywords"]]
df = df.drop_duplicates(subset=["sentence_text","feature_category_insight_keywords","feature_keywords"])

df.to_csv("mdu_tagged_dataset.csv",sep="\t",index=False)
print(df.shape)

def get_tokens(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        tokens.append((token.text, token.tag_, token.dep_, token.head.tag_))
    only_tokens = [t[0] for t in tokens]
    return tokens,only_tokens

def get_sublist(sentence_token_list,keyword_list,opinion_list):

    complete_list = []
    keyword_results=[]
    sll=len(keyword_list)
    for ind in (i for i,e in enumerate(sentence_token_list) if e==keyword_list[0]):
        if sentence_token_list[ind:ind+sll]==keyword_list:
            keyword_results.append((ind,ind+sll-1))

    sll=len(opinion_list)
    opinion_results=[]
    for ind in (i for i,e in enumerate(sentence_token_list) if e==opinion_list[0]):
        if sentence_token_list[ind:ind+sll]==opinion_list:
            opinion_results.append((ind,ind+sll-1))

    if len(keyword_results) == 1 and len(opinion_results) == 1:  
        keyword_start_index = keyword_results[0][0]
        keyword_end_index = keyword_results[0][1]
        opinion_start_index = opinion_results[0][0]
        opinion_end_index = opinion_results[0][1]

        sll = None
        temp_list = None
        if keyword_start_index > opinion_start_index:
            temp_list = sentence_token_list[opinion_start_index:keyword_end_index+1]
        else:
            temp_list = sentence_token_list[keyword_start_index:opinion_end_index+1]

        if temp_list != None:
            sll = len(temp_list)
            for ind in (i for i,e in enumerate(sentence_token_list) if e==temp_list[0]):
                if sentence_token_list[ind:ind+sll]==temp_list:
                    complete_list.append((ind,ind+sll-1))

    return complete_list


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

grp = df.groupby(["sentence_text"])
results = []
sentence_results = []
sent_counter = 1
for sentence, g in grp:

        #if sentence =="Alem and Reese have also been amazing concierges and they have always been super friendly and helpful.":
        grp_keyword = g.groupby(["feature_keywords"])
        grp_opinion = g.groupby(["feature_category_insight_keywords"])

        lst = []
        sent_tokens_details,sent_tokens = get_tokens(sentence)
        offset_list = []
        matched_keyword_index_list = []
        matched_opinion_index_list = []
        single_list = []

        for keyword, opinion_df in grp_keyword:
            keyword_tokens_details,keyword_tokens = get_tokens(keyword)
            opinion_list = opinion_df["feature_category_insight_keywords"].tolist()

            uniq_opinion_list = list(set(opinion_list))
            
            if len(opinion_list) > 1:
                for opinion in opinion_list:
                    opinion_tokens_details,opinion_tokens = get_tokens(opinion)
                    index_list = get_sublist(sent_tokens,keyword_tokens,opinion_tokens)
                    matched_keyword_index_list.extend(index_list)
                lst.append((keyword,opinion_list))

            if len(uniq_opinion_list) == 1:
                opinion_tokens_details,opinion_tokens = get_tokens(opinion_list[0])
                index_list = get_sublist(sent_tokens,keyword_tokens,opinion_tokens)
                single_list.extend(index_list)
                lst.append((keyword,opinion_list))
                
        for opinion, opinion_df in grp_opinion:
            opinion_tokens_details,opinion_tokens = get_tokens(opinion)
            keyword_list = opinion_df["feature_keywords"].tolist()
            if len(keyword_list) > 1:
                for keyword in keyword_list:
                    keyword_tokens_details,keyword_tokens = get_tokens(keyword)
                    index_list = get_sublist(sent_tokens,keyword_tokens,opinion_tokens)
                    matched_opinion_index_list.extend(index_list)
                lst.append((keyword_list,opinion))


        if len(matched_keyword_index_list) > 0 :
            out = [item for t in matched_keyword_index_list for item in t]
            min_value = min(out)
            max_value = max(out)
            offset_list.append((min_value,max_value))

        if len(matched_opinion_index_list) > 0 :
            out = [item for t in matched_opinion_index_list for item in t]
            min_value = min(out)
            max_value = max(out)
            offset_list.append((min_value,max_value))

        index_list = []
        if len(matched_keyword_index_list) > 0 and len(matched_opinion_index_list) > 0:
            index_list = matched_keyword_index_list+matched_opinion_index_list
        elif len(matched_keyword_index_list) > 0:
            index_list = matched_keyword_index_list
        else:
            index_list = matched_opinion_index_list

        clean_list = list(set(offset_list))
        if len(clean_list) == 2:
            list_one = [i for i in range(clean_list[0][0],clean_list[0][1]+1) ]
            list_two = [i for i in range(clean_list[1][0],clean_list[1][1]+1) ]
            commen_element = list(set(list_one).intersection(list_two))
            if len(commen_element) > 0 :
                clean_list = [(clean_list[0][0],clean_list[1][1])]
            else:
                clean_list = offset_list

        if len(single_list) > 0 and len(matched_keyword_index_list) == 0 and len(matched_opinion_index_list) == 0:
            clean_list = []
            out = [item for t in single_list for item in t]
            min_value = min(out)
            max_value = max(out)
            clean_list.append((min_value,max_value))

        sentence = " ".join(sent_tokens)

        entities = []
        phrase_list = []
        sent_tokens_tple_list = [(t[0],t[1],t[2],t[3],'O') for t in sent_tokens_details]

        for tpl in clean_list:
            sublist_tokens = sent_tokens[tpl[0]:tpl[1]+1]
            if len(sublist_tokens) <= 18 :
                sub_string = " ".join(sublist_tokens)
                start_index = sentence.find(sub_string)
                end_index = start_index + len(sub_string)

                sub_string = sentence[start_index:end_index]
                phrase_list.append(sub_string)
                entities.append((start_index,end_index,"KEYWORD"))

                matched_index_list = find_sub_list(sublist_tokens,sent_tokens)
                for tpl in matched_index_list:
                    start_value = tpl[0]
                    end_value = tpl[1]
                    count = 0 
                    for i in range(start_value,end_value+1):
                        tmp = sent_tokens_tple_list[i]
                        if count == 0 :
                            tmp = (tmp[0],tmp[1],tmp[2],tmp[3],'B-KEYWORD')
                            sent_tokens_tple_list[i] = tmp
                            count = count + 1
                        else:
                            tmp = (tmp[0],tmp[1],tmp[2],tmp[3],'I-KEYWORD')
                            count = count + 1
                            sent_tokens_tple_list[i] = tmp

        for value,token in enumerate(sent_tokens_tple_list):
            if value == 0 :
                temp = {
                    "Sentence #":sent_counter,
                    "Word":token[0],
                    "POS":token[1],
                    "DEP":token[2],
                    "HEAD_DEP":token[3],
                    "Tag":token[4]
                }

                sentence_results.append(temp)
            else:
                temp = {
                    "Sentence #":sent_counter,
                    "Word":token[0],
                    "POS":token[1],
                    "DEP":token[2],
                    "HEAD_DEP":token[3],
                    "Tag":token[4]
                }
                sentence_results.append(temp)  

        sent_counter = sent_counter + 1
        if len(entities) > 0: 
            temp = {"sentence":sentence,"phrases":phrase_list,"aspect":lst,"sent_tokens":sent_tokens,"index":clean_list,"matched":index_list, "entities":entities}
            results.append(temp)

print(sent_counter)

df = pd.DataFrame(results)
df.to_csv("qced_dataset.csv",sep="\t", index=False)

df = pd.DataFrame(sentence_results)
df.to_csv("qced_dataset_iob.csv",sep="\t", index=False)
