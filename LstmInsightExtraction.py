import pandas as pd
import os
from MDUExtraction import LSTMFeatureOpinionExtraction
from PhraseExtractionClassification import LSTMPhraseExtraction

current_dir = os.getcwd()

feature_opinion_obj = LSTMFeatureOpinionExtraction.getInstance()
phrase_obj = LSTMPhraseExtraction.getInstance()

df = pd.read_excel(current_dir+"/bert_output.xlsx")
df = df.dropna(subset=["SentenceText"])
df = df.drop_duplicates(subset=["SentenceText"])

print(df.shape)
#df = df.head(5)

#df = pd.DataFrame([{
#    "CONTENT":"The views of the city are great ."
#}])

results = []
complete_results = []
phrase_not_found_comments = []
for index, row in df.iterrows():
    content = row["SentenceText"]

    aspect_keyword_list = feature_opinion_obj.getInsight(content)
    aspect_phrase_list = phrase_obj.getPhrases(content)

    if len(aspect_keyword_list) > 0 and len(aspect_phrase_list) > 0:

        for keyword_record in aspect_keyword_list:
            keword_list = keyword_record["KEYWORD"]
            opinion_list = keyword_record["OPINION"]
            comment = keyword_record["COMMENT"]
            for kewword_dict in keword_list:
                keyword_tokens = kewword_dict["keyword"].split()
                for opinion_dict in opinion_list:
                    opinion_tokens = opinion_dict["keyword"].split()

                    for phrase_record in aspect_phrase_list:
                        phrases = phrase_record["phrases"]
                        for phrase_dict in phrases:
                            phrase_tokens = phrase_dict["phrase"].split()
                            common_feature_tokens = list(set(phrase_tokens).intersection(keyword_tokens))
                            common_opinion_tokens = list(set(phrase_tokens).intersection(opinion_tokens))
                            
                            if len(common_feature_tokens) > 0 and len(common_opinion_tokens) > 0:
                                keyword_text = " ".join([k[1] for k in kewword_dict["tag"]])
                                opinion_text = " ".join([k[1] for k in opinion_dict["tag"]])
                                temp = {
                                    "comment":comment,
                                    "keyword":keyword_text,
                                    "opinion":opinion_text,
                                    "feature_tag":kewword_dict["tag"],
                                    "opinoin_tag":opinion_dict["tag"],
                                }

                                complete_results.append(temp)

        complete_results.extend(results)

if len(complete_results) > 0:
    df_result = pd.DataFrame(complete_results)
    df_result = df_result.drop_duplicates(subset=["comment","keyword","opinion"])
    print(df_result.shape)
    df_result.to_csv(current_dir+"/lstm_result.csv",index=False,sep="\t")
