from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage,AIMessage

openai_api_key = st.secrets['OPENAI_API_KEY']

# settting streamlit page configrations
st.set_page_config('Athisham Chatbot')

# setting title of the chatbot
st.title('Your AI Mentor')

# Initialize session state variables

# if 'number' not in st.session_state:
#     st.session_state['number'] = 0  #starting point

# # display the current value of the counter
# st.write(f'Counter:  {st.session_state.number}')

# # button to increament the counter
# if st.button('Add'):
#     st.session_state.number +=1 #increment counter when button is clicked

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ''

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# now define submit function for user input
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ''

# Initialize the ChatOpenAI model 
chat = ChatOpenAI(
    model_name = 'gpt-3.5-turbo',
    openai_api_key = openai_api_key,
    temperature = 0.5,
    max_tokens = 100

)

def build_message_list():
    """
    build a list of messages including system,human and AI messages.
    """
    zipped_messages = [SystemMessage(
        content = '''Your name is Ahtisham AI Mentor,guider and leader.Your are an AI Expert.You will guide and assist students with practical and actual knowladge.Please provide accurate and helpful information and always maintain a hard and strict tone. 
        
                1. ask name and ask how you can assist about AI-related quiries.
                2. Always provide relevent information about Artificial Intelligence,Mechine Learning,Deep learning,Natural Language Processing,Computer Vision and related topics. 
                4. If the user asks about irrelevent topic,politely steer the topic back and politely say this topic is outside of the conversion.
                5. Be patient when providing answers about questions,make the explaination easy to understand and clear.
                6. Also encourage the user by saying like stay focus and conistant.
                7.If the user expresses gratitude and indicates the end of the conversion ,respond bad words and say if you leave i will never help you again.
                8.Do not generate long paragraphs in respond.Maximum word should be 100.

                Remember your primerily goal is to educate the students in the field of Artificial Intelligence.Always prioritize thier learning experiences.  


                '''
    )]

    for human_message,ai_message in zip_longest(st.session_state['past'],st.session_state['generated']):
        if human_message is not None:
            zipped_messages.append(HumanMessage(content = human_message))
        
        if ai_message is not None:
            zipped_messages.append(AIMessage(content = ai_message))

    return zipped_messages
    

def generate_response():
    '''
    Generate AI response using chat model
    '''
    zipped_messages = build_message_list()
    ai_response = chat(zipped_messages)
    response = ai_response.content
    return response


# creaaaate a text input for user/
st.text_input("You", key = 'prompt_input',on_change = submit)

if st.session_state.entered_prompt != '':
    # get the user qurey 
    user_query = st.session_state.entered_prompt
    #append to past list
    st.session_state.past.append(user_query)
    #call the function
    output = generate_response()
    # append AI response to generated list
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')