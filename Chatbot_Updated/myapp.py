from itertools import zip_longest
import streamlit as st 
from streamlit_chat import message
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage,AIMessage

my_api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config("Ahtisham ChatBot")

st.title("Your Docter")

# if "number" not in st.session_state:
#     st.session_state['number'] = 0

# st.write(f'Counter: {st.session_state.number}')

# if st.button("Add"):
#     st.session_state.number +=1

if 'letest_prompt' not in st.session_state:
    st.session_state['letest_prompt'] = ""

if 'ai_generated' not in st.session_state:
    st.session_state['ai_generated'] = []

# prompt = st.chat_input("Your Prompt")
# if prompt:
#   st.session_state.prompt = prompt

# st.write(f'Current Prompt:"{st.session_state.prompt}"')

if "past_prompts_list" not in st.session_state:
    st.session_state['past_prompts_list'] = []


#Initializing the chat bot model

chat_model = ChatOpenAI(model= 'gpt-3.5-turbo',openai_api_key = my_api_key,temperature = 0.5,max_tokens = 100)

def message_list():
    all_messages = [SystemMessage(content = "You are a helpful assistance for data analysis")]
    for my_prompt,model_reply in zip_longest(st.session_state['past_prompts_list'],st.session_state['ai_generated']):
        if my_prompt:
            all_messages.append(HumanMessage(content = my_prompt))
        if model_reply:
            all_messages.append(AIMessage(content = model_reply))

    return all_messages 


def for_ai_response():
    all_messages = message_list()
    ai_response = chat_model(all_messages)
    ai_response_content = ai_response.content
    return ai_response_content

def submit():
    st.session_state.letest_prompt = st.session_state.input_prompt
    # st.session_state.input_prompt = 
#creating a chat input for users
st.chat_input("Enter Prompt",key = 'input_prompt',on_submit = submit )

if st.session_state.letest_prompt != "":
    #get the user query
    user_query = st.session_state.letest_prompt
    #append the user query to past_prompts_list
    st.session_state.past_prompts_list.append(user_query)
    output = for_ai_response()
    st.session_state['ai_generated'].append(output)

if st.session_state['ai_generated']:
    for i in range(len(st.session_state['ai_generated'])-1,-1,-1):
        message(st.session_state['ai_generated'][i],key = str[i])
        message(st.session_state['past_prompts_list'][i],is_user=True,key = str(i)+"_user")
                   