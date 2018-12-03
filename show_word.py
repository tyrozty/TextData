from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(background_color="white",width=1000, height=860, margin=2)
with open('./res_tfidf_10_class.txt', 'r') as f:
    data = f.readlines()
    class_temp = ''
    class_cnt = 0
    for i in range(len(data) - 1):
        if data[i][1] == data[i+1][1]:
            text = data[i].split(',')[1][:-2]
            class_temp += text
        else:
            wordcloud = wordcloud.generate(class_temp)
            plt.imshow(wordcloud)
            plt.axis("off")
            class_temp = ''
            wordcloud.to_file('./plot_cluster/cluster_%d_res.png' %class_cnt)
            class_cnt += 1

