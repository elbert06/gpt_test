import tensorflow as tf
import numpy as np
import pickle
# from get_country import get_capital
questions=pickle.load(open('questions.pkl','rb'))
# assert 1==2

with open("answers.pkl","rb") as f:
    answers = pickle.load(f)
# print(questions,answers)
# 문자 수준 토크나이저를 사용하여 텍스트를 인덱스로 변환
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=True)
tokenizer.fit_on_texts(list(questions) + list(answers))

# 텍스트를 시퀀스로 변환
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)
# # 시퀀스 패딩
max_len_q = max([len(seq) for seq in question_sequences])
max_len_a = max([len(seq) for seq in answer_sequences])

question_sequences = tf.keras.preprocessing.sequence.pad_sequences(question_sequences, maxlen=max_len_q, padding='post')
answer_sequences = tf.keras.preprocessing.sequence.pad_sequences(answer_sequences, maxlen=max_len_a, padding='post')
# 다양한 길이의 디코더 입력과 타겟 시퀀스 생성
decoder_input_data = []
answer_sequences_target = []
encoder_input_data = []
for question, answer in zip(question_sequences, answer_sequences):
    for i in range(1, len(answer)):
        encoder_input_data.append(question)
        decoder_input_data.append(answer[:i])
        answer_sequences_target.append(answer[1:i+1])
# # 패딩
question_sequences = np.array(encoder_input_data)
answer_sequences_input = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=max_len_a, padding='post')
answer_sequences_target = tf.keras.preprocessing.sequence.pad_sequences(answer_sequences_target, maxlen=max_len_a, padding='post')

# # 타겟 시퀀스를 3차원으로 변환
answer_sequences_target = np.expand_dims(answer_sequences_target, -1)
print(question_sequences.shape,answer_sequences_input.shape,answer_sequences_target.shape)
# # 타겟 시퀀스 준비
vocab_size = len(tokenizer.word_index) + 1
from tensorflow.keras.models import Model,load_model
model = load_model("country.h5")
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
while True:
    # Creating text
    input_text = input("give a chat:")

    question = tokenizer.texts_to_sequences([input_text])
    question = tf.keras.preprocessing.sequence.pad_sequences(question, maxlen=max_len_q, padding='post')
    get_result = reverse_word_map.get
    input_value = [0]*max_len_a
    input_value[0] = tokenizer.word_index.get("<start>")
    print(tokenizer.word_index.get("<start>"))
    sentence = ""
    for k in range(1,answer_sequences_input.shape[1]):
        value_result = list(np.argmax(a) for a in model.predict([question.reshape(1,max_len_q,), np.array(input_value).reshape(1,max_len_a,)],verbose=0)[0])
        value_result = (list(map(get_result, value_result)))
        next_word = value_result[k-1]
        # print(next_word,k-1,value_result)
        if next_word == "<end>":
            break
        elif next_word.__contains__(".") and not any(temp.isdigit() for temp in next_word):
          print(next_word)
        else:
          print(next_word,end=" ")
        sentence += next_word+" "
        input_value[k] = list(reverse_word_map.values()).index(next_word)+1
    # print(sentence.replace("\n",""))