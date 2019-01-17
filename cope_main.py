from __future__ import print_function
import smtplib #to send emails

import email_config
from google.cloud import language
import os
import boto3
import logging
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import networkx as nx
import matplotlib.pyplot as plt
import remove_words
import csv
import nltk
from nltk.corpus import wordnet


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
dynamoTable_error = dynamodb.Table('build_model_error')
dynamoTable_success = dynamodb.Table('build_model_success')

session_attributes = {}
answer_list = {}
    
syn_dict = remove_words.syn_dict_word_net;

error_log = []

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./apikey.json"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

nltk.data.path.append("/var/task/data_nltk/nltk_data"); #pointing the word net data to the given path so that it works in aws

#-------------------------------------speech templates starts--------------------------------------------------------------
SKILL_NAME = "Build a Model"

WELCOME_MESSAGE ="Hi welcome to "+SKILL_NAME+"! We will be building a causal model for a problem of your choice. I will ask you which concepts drive this problem. Then, we will look at the drivers on these concepts. For more information regarding "+SKILL_NAME+" you can say 'Alexa Help me'. If you are ready to begin, just say begin."
HELP_MESSAGE = """"Let's """+SKILL_NAME+"""" is a skill that creates causal models out of your answers. 
    For example if you wish to make a causal model of obesity, questions will be asked such as "what do you think caues obesity?". 
    I will identify the main factors in your answers, and build on these to ask more questions. 
    Once all the questions are answered, I will create a causal map, and email to the registered email id with Amazon.
    In short, we're going to have a conversation, and I will track your thoughts by building a causal map. 
    Can we continue?"""

HELP_REPROMPT = "All you have to do is answer to my questions in a single word or a small phrase with your answers I will create causal map and email it to your registered email id with amazon. Ready to resume where you left off?"

STOP_MESSAGE = 'Thank you for your interest in using '+SKILL_NAME+'. Have a nice day!'

NO_OTHER_entities = 'Thank you for answering all the questions. The causal map will be mailed to your email ID';
ERROR_MESSAGE ='There has occured an error';

BEGIN_MESSAGE = 'Ok, let\'s begin. ';
CAPTURE_ENTITY = 'We need a core concept for your map. For instance, you could create a map about obesity or homelessness. What would you like the core concept of your map to be?. For example, you can say \'I would like the core concept to be...\'';

ROUND_2_QUESTION = ' Now we will discuss the causes of '
ROUND_2_QUESTION_FOLLOW_UP = 'Can you name all the causes of '
ROUND_2_QUESTION_FOLLOW_UP_ANOTHER = 'Can you name another cause of ';
ROUND_2_QUESTION_FOLLOW_UP_1 = ' Go ahead, name all the causes for '
ROUND_2_ENTITIES_IDENTIFIED = 'You have provided all the causes of '

ROUND_2_DELETE_ENTITIES = ' If there are any causes that you\'d like me to remove, please say remove followed by the word.'
ROUND_2_REMOVED_ENTITIES = 'has been removed. ';

client = language.LanguageServiceClient()

#-------------------------------------speech templates ends--------------------------------------------------------------


# --------------- Helpers that build all of the responses ----------------------


def build_speechlet_response(title, output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },                                                                                                                                                                                                                              
        'card': {
            'type': 'Simple',
            'title': "SessionSpeechlet - " + title,
            'content': "SessionSpeechlet - " + output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session,        
    }


def build_response(session_attributes, speechlet_response):
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': speechlet_response
    }


#-----------------functions -----------------------------#

def and_string_fun(entities_list):
    and_text = ''
    print("and_string_fun entities_list:",entities_list)
    print("len(entities_list):",len(entities_list))
    if(len(entities_list) == 1):
        for i in entities_list:
            and_text = i
    else:
        for i in range(0, len(entities_list)-1):
            and_text+= entities_list[i]+", "
        and_text +=" and "+ entities_list[len(entities_list)-1]

    return and_text

#----------------custom functions------------------------#


# --------------- Functions that control the skill's behavior ------------------#
def get_welcome_response():
    session_attributes.clear()
    print("entered Launch/get_welcome_message_response intent")
    card_title = SKILL_NAME
    print("card_title:",card_title)
    session_attributes.update({
        'yes_begin_skill':True, 
        'no_begin_skill':True,
        'del_rel_form':{}, 
        'question':[],
        'intent_triggered':[],
        'answer':[], 
        'current_question_word':'',
        'stop_layer':False, 
        'help_intent':False,
        'identified_entities_copy':{}
    })
    should_end_session = False    
    speech_output = WELCOME_MESSAGE   
    reprompt_text = WELCOME_MESSAGE
    session_attributes.update({'repeat':speech_output})
    question = []
    question.append("Welcome to "+SKILL_NAME+". If you are ready to begin, just say begin.")
    session_attributes.update({'question':question})      
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))    

def handle_session_end_request(intent, session, proceed):  
     
    answer = session['attributes']['answer']

    if(proceed == True):
        answer.append("Proceed")
    else:
        answer.append("No")

    session_attributes.update({'answer': answer})
    
    question = session['attributes']['question']
    
    card_title = intent['name']

   
    if(session['attributes']['no_begin_skill'] == True) or (session['attributes']['help_intent'] == True):
        print("entered no I do not want to answer the questions intent")
        card_title = STOP_MESSAGE
        session_attributes.update({'yes_begin_skill':False, 'no_begin_skill':False, 'help_intent':False})
        should_end_session = True
        speech_output = STOP_MESSAGE
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, None, should_end_session))

    else: 
        entities_count =  session['attributes']['entities_count']
        first_layer_entities = session['attributes']['first_layer_entities']
        identified_entities = session['attributes']['identified_entities']
        identified_entities_copy = session['attributes']['identified_entities_copy']
        current_question_word = session['attributes']['current_question_word']
        print("Second round entered")
        current_layer_number = session['attributes']['current_layer_number']
        # if its the end of all the entities extracted
        #Prints the list of all the entities, builds a map, email's the map to the user, updates the log to dynamo db

        if(entities_count == len(first_layer_entities)-1):
            print("entered entities_count == len(first_layer_entities)")
            card_title = SKILL_NAME+" ended"
            should_end_session = True
            and_string = ''

            if(len(first_layer_entities) == 1):
                for i in first_layer_entities:
                    speech_output = "You have not provided me any cause for "+ first_layer_entities[0] +".Please try again from the start."
                    reprompt_text = speech_output 
                    update_into_table(session, error = False)
                    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

            else:
                #succefully answered all the questions
                speech_output = NO_OTHER_entities
                reprompt_text = speech_output
                identified_entities_combined = combine_dict(identified_entities, identified_entities_copy)    
                create_a_map(identified_entities_combined)
                send_email('datalab.science@gmail.com')
                send_log_to_the_developer_when_skill_breaks(session,error=False)
                update_into_table(session, error = False)
                return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

            

        #if its not the end of all the entities extracted then ask more questions 

        else:
            print("not the end of all the entities extracted then ask more questions")
            print("current_question_word="+current_question_word)
            print("identified_entities.keys()="+','.join(identified_entities.keys()))
            current_layer_number = session['attributes']['current_layer_number']
            no_more_entities = session['attributes']['no_more_entities']

            #answer entites for a question is over
            for x in identified_entities:
                if(x == current_question_word):
                    print(x +"=="+ current_question_word)
                    if((identified_entities[x] != []) and (no_more_entities == True) and (current_question_word != identified_entities[list(identified_entities.keys())[entities_count]][len(identified_entities[list(identified_entities.keys())[entities_count]])-1])): 
                        print("entered len(identified_entities[{}]) != 0)".format(x)) 
                        should_end_session = False
                        #ask for the entities if they have to be removed
                        and_string = ''
                        if(len(identified_entities[current_question_word]) == 1):
                            for i in identified_entities[current_question_word]:
                                and_string = i
                        else:
                            for i in range(0, len(identified_entities[current_question_word])-1):
                                and_string+= identified_entities[current_question_word][i]+", "
                            and_string +=" and "+ identified_entities[current_question_word][len(identified_entities[current_question_word])-1]   
                        speech_output = ROUND_2_ENTITIES_IDENTIFIED + current_question_word +". I took note of the following causes: {} ".format(and_string)+"."+ROUND_2_DELETE_ENTITIES+', If you don\'t want to discuss more distal causes, you can say "stop". To continue, please say "proceed".'
                        reprompt_text = speech_output 
                        if first_layer_entities[entities_count+1] not in identified_entities.keys():
                            for x in range(entities_count,len(first_layer_entities)-entities_count):
                                print("first_layer_entities[x]:",first_layer_entities[x])
                                if first_layer_entities[x] not in identified_entities.keys():
                                    identified_entities.update({first_layer_entities[x]:[]})    
                       
                        question.append(reprompt_text)
                        print("about to enter current_question_word not in identified_entities")
                        session_attributes.update({
                            'question':question, 
                            'repeat':speech_output,
                            'identified_entities': identified_entities,
                            'current_question_word':first_layer_entities[entities_count],
                            'no_more_entities':False,
                            'entities_count': entities_count,
                            'stop_layer':True,
                        })
                        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

                    else:
                        end_layer_pos = session['attributes']['end_layer_pos']
                        current_layer_starting_pos = session['attributes']['current_layer_starting_pos']
                        current_layer_number = session['attributes']['current_layer_number']

                        if(current_question_word == first_layer_entities[current_layer_starting_pos]):
                            end_layer_pos = len(first_layer_entities)
                            session_attributes.update({
                                'end_layer_pos':end_layer_pos,
                                'current_layer_starting_pos':current_layer_starting_pos})
                        else:
                            if((entities_count+1) == end_layer_pos):
                                print("entered the else section ")
                                print("Level omplete")
                                should_end_session = False
                                speech_output = "You have completed answering questions to "+first_layer_entities[current_layer_starting_pos]+" and its causes. If you would like to end your session here, you can do that by saying 'STOP' or if you would like to answer next layer of questions, please say 'proceed"
                                reprompt_text = speech_output
                                question.append(reprompt_text)
                                current_layer_starting_pos = end_layer_pos
                                end_layer_pos = len(first_layer_entities)
                                current_layer_number = current_layer_number + 1
                                session_attributes.update({
                                    'no_more_entities':False,
                                    'question':question,
                                    'repeat':speech_output,
                                    'end_layer_pos':end_layer_pos,
                                    'entities_count':entities_count,
                                    'current_layer_starting_pos':current_layer_starting_pos,
                                    'current_layer_number':current_layer_number
                                })                   
                                return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
                             


#--------------------------------------------end of completed answering questions for a single layer-----------------------------------------------------------

                        print("entered the else else section ")
                        current_question_word = first_layer_entities[entities_count]
                        entities_count = entities_count + 1
                        should_end_session = False
                        speech_output = ROUND_2_QUESTION_FOLLOW_UP + first_layer_entities[entities_count] + " that you're aware of ?"
                        reprompt_text = speech_output
                        current_question_word = first_layer_entities[entities_count]
                        identified_entities[current_question_word] = []
                        identified_entities.update(identified_entities[current_question_word])
                        question.append(reprompt_text)
                        session_attributes.update({'entities_count':entities_count,
                            'round_2':True,
                            'no_more_entities':False,
                            'current_question_word':current_question_word,
                            'identified_entities':identified_entities,
                            'question':question,
                            'repeat':speech_output
                        })                   
                        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))    
  
    

def yes_intent_request(intent, session):
    
    answer = session['attributes']['answer']
    answer.append("Yes")
    session_attributes.update({'answer': answer})

    if(session['attributes']['help_intent'] == True):
        return get_welcome_response()

    question = session['attributes']['question']
    if(session['attributes']['yes_begin_skill'] == True):
        print("entered yes intent")
        card_title = intent['name']
        speech_output = BEGIN_MESSAGE + CAPTURE_ENTITY
        reprompt_text = CAPTURE_ENTITY 
        question.append(reprompt_text)       
        should_end_session = False
        session_attributes.update({
            'yes_begin_skill':False, 
            'no_begin_skill':False, 
            'round_1':True,
            'entities_count':0, 
            'first_layer_entities':[],
            'identified_entities':{},
            'no_more_entities':True,
            'repeat':speech_output,
            'question':question,
            'stop_layer':False,
            'end_layer_pos':0,
            'current_layer_starting_pos':0,
            'current_layer_number':1
        })
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
    else:
        
        identified_entities = session['attributes']['identified_entities']
        print("entered yes I can name another entity intent")
        entities_count = session['attributes']['entities_count']
        first_layer_entities = session['attributes']['first_layer_entities']
        current_layer_number = session['attributes']['current_layer_number']
        card_title = intent['name']   
        speech_output = ROUND_2_QUESTION_FOLLOW_UP_1 + first_layer_entities[entities_count]+'.'
        reprompt_text = speech_output  
        should_end_session = False
        question.append(reprompt_text)
        session_attributes.update({
            'question':question,
            'repeat':speech_output
        })
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
    

def proceed_intent_request(intent, session):
    print("sending the pointer to no entity")
    return handle_session_end_request(intent, session, proceed = True)

def remove_intent_request(intent, session):
    print("entered remove intent")
    card_title = intent['name']

    should_end_session = False
    
    first_layer_entities = session['attributes']['first_layer_entities']
    entities_count = session['attributes']['entities_count']
    current_layer_number = session['attributes']['current_layer_number']
    current_question_word = session['attributes']['current_question_word']
    identified_entities = session['attributes']['identified_entities']
    answer = session['attributes']['answer']
    question = session['attributes']['question']
    del_rel_form = session['attributes']['del_rel_form']
    if(len(current_question_word)>1):
        if((current_question_word in identified_entities.keys()) and len(identified_entities.keys()) > 0):
            identified_entities_list = identified_entities[current_question_word]
        else: identified_entities_list = []
    else: identified_entities_list = []
    
    current_question_word = first_layer_entities[entities_count]
    identified_entities_list = identified_entities[current_question_word]
    
    answer_from_user = intent['slots']['remove_entity']['value']
    response = fetch_entities_from_google_api(answer_from_user)
    remove_entities_list = []
    for entity in response.entities:  
        print("entity.name:",entity.name)          
        if entity.name in first_layer_entities:
            remove_entities_list.append(entity.name)
            print("remove_entities_list:"+','.join(remove_entities_list))

    print("remove_entities_list:", remove_entities_list)

    for x in remove_entities_list:
        if x in first_layer_entities:
            first_layer_entities.remove(x)
            identified_entities_list.remove(x)
            if x in identified_entities.keys():
                del identified_entities[x]
            del del_rel_form[x]
    identified_entities[current_question_word] = identified_entities_list

    session_attributes.update({'identified_entities':identified_entities})
    current_layer_number = session['attributes']['current_layer_number']
   
    and_text = and_string_fun(remove_entities_list)
    if(identified_entities[current_question_word] != []):
        and_text_remaining = and_string_fun(identified_entities[current_question_word])
        speech_output = and_text +" has been removed . The remaining causes for "+ current_question_word+ " include "+and_text_remaining+". If you would like to remove any other causes, Please say remove followed by the word, If not say 'proceed' to continue"
    else:
        speech_output = and_text +" has been removed. You have not provided any cause for "+ current_question_word+ ". Say 'proceed' to continue or 'Stop' to stop here"
    
    reprompt_text = speech_output
    question.append(reprompt_text)
    session_attributes.update({
        'first_layer_entities':first_layer_entities,
        'identified_entities':identified_entities,
        'entities_count':entities_count,
        'current_question_word':first_layer_entities[entities_count],
        'question':question,
        'repeat':speech_output,
        'del_rel_form':del_rel_form
    })

    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))


def model_creator_intent(intent, session):
    
    first_layer_entities = session['attributes']['first_layer_entities']
    entities_count = session['attributes']['entities_count']    
    current_question_word = session['attributes']['current_question_word']
    identified_entities = session['attributes']['identified_entities']
    answer = session['attributes']['answer']
    question = session['attributes']['question']
    del_rel_form = session['attributes']['del_rel_form']
    identified_entities_copy = session['attributes']['identified_entities_copy']

    if(len(current_question_word)>1):
        if((current_question_word in identified_entities.keys()) and len(identified_entities.keys()) > 0):
            identified_entities_list_extenion = list(set(identified_entities[current_question_word]))
        else:
            identified_entities_list_extenion = []
    else:
        identified_entities_list_extenion = []

    print("identified_entities_list_extenion:",identified_entities_list_extenion)
    print("entered model_creator_intent_request")

    card_title = intent['name']
    
    should_end_session = False

    answer_from_user = intent['slots']['user_causes']['value']

    if(answer_from_user == ''):
        speech_output = "Sorry,I was unable to capture the entites in your answer. Please repeat your answer, clearly"
        reprompt_text = speech_output
        question.append(reprompt_text)
        session_attributes.update({
                'question':question,
                'repeat':speech_output,
            })
        
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

    answer.append(answer_from_user)
    print("answer_from_user:"+answer_from_user)
    
    session_attributes.update({ 
        'no_begin_skill':False,
        'answer':answer,
        'no_more_entities':True       
    })
    word_list = list(set(answer_from_user.split()))
    answer_from_user = ' '.join([i for i in word_list if i not in ["core","map","concept"]])
    print("answer_from_user after removal:",answer_from_user)
    #-------------------google API--------------------------#
    response = fetch_entities_from_google_api(answer_from_user)
    print("response:",response)
    #-------------------google API--------------------------#

    entities_count = session['attributes']['entities_count']
    identified_entities_list =[]
    current_entities = []
    if(len(question)==2):
         # capture the core concept if its not captured yet
        for entity in response.entities: 
            if(entity.name not in syn_dict):
                print("entity.name:",entity.name)
                if(len((entity.name).split()) > 2):
                    entities_split = (entity.name).split()
                    for x in entities_split:
                        identified_entities_list.append(x)
                else:
                    identified_entities_list.append(entity.name)
                    first_layer_entities.append(entity.name)

        # if(identified_entities_list == []):
        #     identified_entities_list.append("overeating")
        #     first_layer_entities.append("overeating")
        # if the named core concept is a single concept
        if(len(identified_entities_list) == 1):
            identified_entities.update({identified_entities_list[0]:[]})
            first_layer_entities = remove_duplicates(first_layer_entities)
            print("first_layer_entities remove duplicate:",first_layer_entities)
            speech_output = "Great, we'll make a map for "+identified_entities_list[0]+"."+ROUND_2_QUESTION + identified_entities_list[0] + ". Please tell me which causes you think of. For example, you can start saying: I think it is caused by..."
            reprompt_text = speech_output
            question.append(reprompt_text)
            del_rel_form = derivationally_related_form(identified_entities_list, del_rel_form)
            session_attributes.update({
                    'identified_entities': identified_entities,
                    'first_layer_entities':first_layer_entities,
                    'question':question,
                    'repeat':speech_output,
                    'current_question_word':identified_entities_list[0],
                    'del_rel_form':del_rel_form,
                    'no_more_entities':True 
                })
            
            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

        # if the named core concept is not a single concept
        elif(len(identified_entities_list) > 1):
            
            first_layer_entities = remove_duplicates(first_layer_entities)
            and_string = and_string_fun(identified_entities_list)
            speech_output = "You have given "+str(len(identified_entities_list))+" core concepts, Which is "+and_string+". You are allowed to form a map with just a single concept.  What would you like the core concept of your map to be?."
            reprompt_text = speech_output
            question.append(reprompt_text)
            session_attributes.update({
                    'repeat':speech_output,
                    'no_more_entities':True 
                })
            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

    #if the core concept has been captured        
    else:
        entities_list = []    
        for entity in response.entities:
            if(len((entity.name).split()) > 1):         
                entities_split = (entity.name).split()
                for x in entities_split:
                    if(x not in syn_dict):
                        entities_list.append(x)
            else:
                if(entity.name not in syn_dict):
                    entities_list.append(entity.name)

        print("entities_list:",entities_list)

        combined_del_rel_form_values = []
        for x in del_rel_form.values():
            combined_del_rel_form_values = combined_del_rel_form_values + x

        for values_entities_list in entities_list:
            print("values_entities_list:",values_entities_list)
            if(values_entities_list in first_layer_entities) and (values_entities_list not in identified_entities[list(identified_entities.keys())[entities_count]]) and (values_entities_list != current_question_word):
                identified_entities_list.append(values_entities_list)
                print("identified_entities_list:",identified_entities_list)
                print("Hi")
            #if captured entity is not an captured entity and it is not currently an entity of the current_question_word
            elif(values_entities_list not in first_layer_entities) and (values_entities_list not in identified_entities[list(identified_entities.keys())[entities_count]]) and (values_entities_list != current_question_word):                    
                print("hello")
                if(values_entities_list not in combined_del_rel_form_values):
                    first_layer_entities.append(values_entities_list)
                identified_entities_list.append(values_entities_list)
                print("first_layer_entities:",first_layer_entities)
                print("identified_entities_list:",identified_entities_list)

    print("identified_entities_list final:",identified_entities_list)

    #----------Start of Check if any entities captured in this round already exists in the values of already captured derivational values---------------#
    already_exist_in_del_rel_form = {}
    identified_entities_list_copy = identified_entities_list
    for values_identified_entities_list in identified_entities_list:
        for name, value in del_rel_form.items():
            if(values_identified_entities_list in value):
                already_exist_in_del_rel_form.update({name:values_identified_entities_list})
                identified_entities_list_copy.remove(values_identified_entities_list)

    identified_entities_list = identified_entities_list_copy

#----------End of Check if any entities captured in this round already exists in the values of already captured derivational values---------------#
    print("identified_entities_list_check1:",identified_entities_list)

    del_rel_form = derivationally_related_form(identified_entities_list, del_rel_form)
    
#----------Start of Check if any entities captured in this round already exists in the values of already captured derivational values---------------#
    
    identified_entities_list_copy = identified_entities_list
    for values_identified_entities_list in identified_entities_list:
        for name, value in del_rel_form.items():
            if(values_identified_entities_list in value):
                already_exist_in_del_rel_form.update({name:values_identified_entities_list})
                identified_entities_list_copy.remove(values_identified_entities_list)

    identified_entities_list = identified_entities_list_copy
    
#----------End of Check if any entities captured in this round already exists in the values of already captured derivational values---------------#

    print("identified_entities_list_check2:",identified_entities_list)

    #finally updates identified_entities list
    for value in identified_entities_list:
        identified_entities_list_extenion.append(value)

    identified_entities[current_question_word] = identified_entities_list_extenion

    for x in identified_entities_list:
        if x not in first_layer_entities:
            identified_entities[x] = []

    session_attributes.update({'identified_entities': identified_entities})
    print("identified_entities:",identified_entities)

    #update identified_entities_copy with already_exist_in_del_rel_form entities 
    if(len(already_exist_in_del_rel_form)>1):
        identified_entities_copy_list = already_exist_in_del_rel_form[current_question_word]
        for x in already_exist_in_del_rel_form:
            identified_entities_copy_list.append(x)
           
        identified_entities_copy[current_question_word] = list(identified_entities_copy_list)
        session_attributes.update({identified_entities_copy:identified_entities_copy})

#--------start of if any current capture entity already exists in del_rel_form------------------------------------------------------------------------#
    if(len(already_exist_in_del_rel_form)>0):
        first_layer_entities = remove_duplicates(first_layer_entities)
        card_title = "entities spoken already exist"
        relation = []
        print("already_exist_in_del_rel_form:",already_exist_in_del_rel_form)
        for name, value in already_exist_in_del_rel_form.items():
            relation.append(value +" is a synonym of "+name)
            l1= []
            l1.append(name)
            identified_entities_copy.update({current_question_word:l1})

        print("identified_entities_copy after update:",identified_entities_copy)

        and_string = and_string_fun(relation)
        should_end_session = False
        speech_output = "It looks like "+and_string+", hence I have saved the synonym instead. "+ROUND_2_QUESTION_FOLLOW_UP_ANOTHER+ first_layer_entities[entities_count] +" that you're aware of ?"
        reprompt_text = speech_output
        repeat = speech_output
        question.append(reprompt_text)
        session_attributes.update({
                'identified_entities': identified_entities,
                'first_layer_entities':first_layer_entities,
                'identified_entities_copy':identified_entities_copy,
                'question':question,
                'repeat':speech_output,
                'no_more_entities':True,
                'repeat':repeat,
                'del_rel_form':del_rel_form

            })
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
#--------end of if any current capture entity already exists in del_rel_form------------------------------------------------------------------------#

    

#-------------if the core concept has already been captured, ask questions--------------------------------------------------#  
    


    first_layer_entities = remove_duplicates(first_layer_entities) 
    speech_output = ROUND_2_QUESTION_FOLLOW_UP_ANOTHER + first_layer_entities[entities_count] +" that you're aware of ?"
    reprompt_text = speech_output
    question.append(reprompt_text)
    session_attributes.update({
            'identified_entities': identified_entities,
            'first_layer_entities':first_layer_entities,
            'question':question,
            'repeat':speech_output,
            'no_more_entities':True,
            'current_question_word':first_layer_entities[entities_count],
            'del_rel_form':del_rel_form
        })
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

def help_intent_request(intent, session):
    
    identified_entities = session['attributes']['identified_entities']
    print("entered help intent")
    card_title = intent['name']
    session_attributes.update({'yes_begin_skill':False, 'no_begin_skill':True, 'help_intent':True})
    should_end_session = False
    speech_output = HELP_MESSAGE
    reprompt_text = HELP_MESSAGE
    session_attributes.update({'repeat':speech_output})
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
    

def fall_back_intent(intent, session):
    identified_entities_copy = session['attributes']['identified_entities_copy']
    identified_entities = session['attributes']['identified_entities']
    card_title = intent['name']
    should_end_session = False
    speech_output = "Please answer in a proper sentence and not just words. Try using sentence starters to answer the questions"
    reprompt_text = speech_output
    send_log_to_the_developer_when_skill_breaks(session,error=True)
    identified_entities_combined = combine_dict(identified_entities, identified_entities_copy)    
    create_a_map(identified_entities_combined)
    send_email('datalab.science@gmail.com')
    update_into_table(session, error = True)
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

def stop_intent_request(intent, session):
    
    answer = session['attributes']['answer']
    answer.append("Stop")
    session_attributes.update({'answer': answer})
    stop_layer = session['attributes']['stop_layer']
    entities_count =  session['attributes']['entities_count']
    current_layer_number = session['attributes']['current_layer_number']
    first_layer_entities = session['attributes']['first_layer_entities']
    identified_entities = session['attributes']['identified_entities']
    identified_entities_copy = session['attributes']['identified_entities_copy']       
    
    if(session['attributes']['stop_layer'] == True):
        print("Stop I do not want to enter next layer")

        # if its the end of all the entities extracted
        #Prints the list of all the entities, builds a map, email's the map to the user, updates the log to dynamo db

        card_title = SKILL_NAME+" ended"
        should_end_session = True
        first_layer_entities_string = ''
        if(len(first_layer_entities) == 1):
            for i in first_layer_entities:
                speech_output = "You have not provided me any cause for "+ first_layer_entities[0] +".Please try again from the start."
                reprompt_text = speech_output
                update_into_table(session, error = False)
                return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

        else:
            # for i in range(1, len(first_layer_entities)-1):
            #     first_layer_entities_string+= first_layer_entities[i]+", "
            # first_layer_entities_string +=" and "+ first_layer_entities[len(first_layer_entities)-1]

            # speech_output = NO_OTHER_entities+'I noted that the causes of '+ first_layer_entities[0] +' are '+first_layer_entities_string+" .Results and the causal map will be mailed to your email ID"
            speech_output = NO_OTHER_entities
            reprompt_text = speech_output 
           
            identified_entities_combined = combine_dict(identified_entities, identified_entities_copy)    
            create_a_map(identified_entities_combined)
            send_email('datalab.science@gmail.com')
            update_into_table(session, error = False)
            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
    else:
        print("entered no I do not want to answer the questions intent")
        card_title = STOP_MESSAGE
        session_attributes.update({'yes_begin_skill':False, 'no_begin_skill':False} )
        should_end_session = True
        speech_output = STOP_MESSAGE
        update_into_table(session, error = False)
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, None, should_end_session))


def error_intent(intent, session):

    print("entered error intent")
    card_title = 'Error occured'    
    should_end_session = True
    speech_output = ERROR_MESSAGE
    update_into_table(session, error = True)
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, None, should_end_session))

def repeat_intent_request(intent, session):
    card_title = intent['name']
    repeat = session['attributes']['repeat']
    should_end_session = False
    speech_output = repeat 
    reprompt_text = repeat
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

#------------------other fucntions--------------------------------#


    






def derivationally_related_form(identified_entities_list, del_rel_form):

    #------------start of adding the synonyms of new words captured ------------------------------------------------------------------------------------------------

    for each_word in identified_entities_list:
        del_rel_form_list = set()
        if(each_word not in del_rel_form.values()) and (each_word not in del_rel_form.keys()):
            for each_synsets in wordnet.synsets(each_word):
                for each_lemma in each_synsets.lemmas():
                    der_name = ''
                    for each_der in each_lemma.derivationally_related_forms():
                        der_name = each_der.name()
                        if('_' in der_name):
                            der_name = ' '.join(der_name.split('_'))
                        for name, value in del_rel_form.items():
                            if(der_name != each_word) and (der_name != ''):    
                                del_rel_form_list.add(der_name)
                del_rel_form[each_word] = list(del_rel_form_list)

    print("del_rel_form:",del_rel_form)
    return del_rel_form

    #------------end of adding the synonyms of new words captured ------------------------------------------------------------------------------------------------   

def try_block_error(session):
    should_end_session = True
    identified_entities = session['attributes']['identified_entities']
    identified_entities_copy = session['attributes']['identified_entities_copy']
    speech_output = "There has occured an error but the answers have been recorded and mailed to you"
    reprompt_text = speech_output
    send_log_to_the_developer_when_skill_breaks(session, error=True)

    identified_entities_combined = combine_dict(identified_entities, identified_entities_copy)    
    create_a_map(identified_entities_combined)
    send_email('datalab.science@gmail.com')
    update_into_table(session,error = True)
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))


# ---------fecth the entites from google api----------------
def fetch_entities_from_google_api(answer_from_user):

        document = language.types.Document(
            content=answer_from_user,
            language='en',
            type='PLAIN_TEXT',
        )
        response = client.analyze_entities(document=document, encoding_type='UTF32',)
        return response

def combine_dict(identified_entities,identified_entities_copy):
    if(identified_entities_copy != {}):
        for key1, value1 in identified_entities.items():
            for key2, value2 in identified_entities_copy.items():
                if(key1 == key2):
                    identified_entities[key1]  = list(set(value1).union(set(value2)))
        return identified_entities
    else:
        return identified_entities

def update_into_table(session, error):
    print("entered update_into_table")
    question = session['attributes']['question']
    answer = session['attributes']['answer']
    identified_entities = session['attributes']['identified_entities']
    identified_entities_copy = session['attributes']['identified_entities_copy']
    identified_entities_combined = combine_dict(identified_entities, identified_entities_copy)
    intent_triggered = session['attributes']['intent_triggered']
    if(error == True):
        intent_triggered.append("error_has_occured")
        dynamoTable_error.put_item(
            Item={
            'userId':session['sessionId'],
            'question':question,
            'intent_triggered':intent_triggered,
            'answer':answer,
            'identified_entities':identified_entities_combined
       })
    else:
        dynamoTable_success.put_item(
            Item={
            'userId':session['sessionId'],
            'question':question,
            'intent_triggered':intent_triggered,
            'answer':answer,
            'identified_entities':identified_entities_combined
       })
    

   

def remove_duplicates(first_layer_entities):
    first_layer_entities_list = []
    for x in first_layer_entities:
        count = 0
        for y in first_layer_entities:
            if(x == y):
                count = count + 1
                if(count == 1) and (y not in first_layer_entities_list):
                    first_layer_entities_list.append(x)            
    first_layer_entities = first_layer_entities_list
    return first_layer_entities   
  
def create_a_map(identified_entities):
    plt.clf()
    fig_size = plt.rcParams["figure.figsize"]
    G=nx.DiGraph()
    csv_filename = '/tmp/node_relations.csv'
    with open(csv_filename, 'w') as f:
        wtr = csv.writer(f, delimiter=',')
        header = ["To","From"]
        wtr.writerow(i for i in header)
        
        #---------increase the image size -----------------#
        len_iden = len(identified_entities.keys())
        
        if(len_iden>=10 and len_iden<=20):
            plt.figure(figsize=(20,8))
        elif(len_iden>=20):
            print("entered len>20")
            plt.figure(figsize=(25,10))
        #---------increase the image size -----------------#

        for key in identified_entities.keys():       
            for z in range(0,len(identified_entities[key])):
                G.add_edges_from([(str(identified_entities[key][z]),str(key))])
                wtr.writerow([key,identified_entities[key][z]])
    f.close()
    nx.draw(G, with_labels=True, font_size=10, node_color='yellowgreen', pos=nx.spring_layout(G),  node_size=2000)
    plt.savefig("/tmp/networkx.png")
    plt.close("/tmp/networkx.png")
    G.clear()



# def get_user_emailId():   

#     if(access_token != None): 
#     #print access_token
#         amazonProfileURL = 'https://api.amazon.com/user/profile?access_token='
#         r = requests.get(url=amazonProfileURL+access_token)
#         if r.status_code == 200:
#             user_details = r.json()
#         else:
#             user_details = False
#             return False
#             print("user_details not found from access token")
#     else:
#         return False

#     if(user_details != False):        
#         user_emailid = user_details['email']
#         return user_emailid
#     else:
#         return False

def send_email(user_emailid):
    body = ''
    def attachment_a_file(filename):
        filename=filename
        attachment = open(filename, 'rb')

        part = MIMEBase('application','octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',"attachment; filename= "+filename)
        msg.attach(part)

    email_user = email_config.EMAIL_ADDRESS
    email_password = email_config.PASSWORD
    email_send = user_emailid

    subject = 'Thank you for using Build a Model - Here are your results'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject


    with open('/tmp/node_relations.csv') as input_file:
        reader = csv.reader(input_file)
        data = list(reader)
    coreConcept =  str(data[1][0])
    mylist = data
    mylistitem = (["""<td style="text-align: left; display:block">""" + str(",".join(i)) + """</td>""" for i in mylist])
    merged = [item for sublist in zip(mylistitem) for item in sublist]
    htmlline = '\n'.join(merged)

    # with open("email_template.html", "r", encoding='utf-8') as f:
    #         body += f.read()
    body = """
    <!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
    <meta charset="utf-8"> <!-- utf-8 works for most cases -->
    <meta name="viewport" content="width=device-width"> <!-- Forcing initial-scale shouldn't be necessary -->
    <meta http-equiv="X-UA-Compatible" content="IE=edge"> <!-- Use the latest (edge) version of IE rendering engine -->
    <meta name="x-apple-disable-message-reformatting">  <!-- Disable auto-scale in iOS 10 Mail entirely -->
    <title></title> <!-- The title tag shows in email notifications, like Android 4.4. -->

    <!-- Web Font / @font-face : BEGIN -->
    <!-- NOTE: If web fonts are not required, lines 10 - 27 can be safely removed. -->

    <!-- Desktop Outlook chokes on web font references and defaults to Times New Roman, so we force a safe fallback font. -->
    <!--[if mso]>
        <style>
            * {
                font-family: sans-serif !important;
            }
        </style>
    <![endif]-->

    <!-- All other clients get the webfont reference; some will render the font and others will silently fail to the fallbacks. More on that here: http://stylecampaign.com/blog/2015/02/webfont-support-in-email/ -->
    <!--[if !mso]><!-->
    <!-- insert web font reference, eg: <link href='https://fonts.googleapis.com/css?family=Roboto:400,700' rel='stylesheet' type='text/css'> -->
    <!--<![endif]-->

    <!-- Web Font / @font-face : END -->

    <!-- CSS Reset : BEGIN -->
    <style>

        /* What it does: Remove spaces around the email design added by some email clients. */
        /* Beware: It can remove the padding / margin and add a background color to the compose a reply window. */
        html,
        body {
            margin: 0 auto !important;
            padding: 0 !important;
            height: 100% !important;
            width: 100% !important;
        }

        /* What it does: Stops email clients resizing small text. */
        * {
            -ms-text-size-adjust: 100%;
            -webkit-text-size-adjust: 100%;
        }

        /* What it does: Centers email on Android 4.4 */
        div[style*="margin: 16px 0"] {
            margin: 0 !important;
        }

        /* What it does: Stops Outlook from adding extra spacing to tables. */
        table,
        td {
            mso-table-lspace: 0pt !important;
            mso-table-rspace: 0pt !important;
        }

        /* What it does: Fixes webkit padding issue. Fix for Yahoo mail table alignment bug. Applies table-layout to the first 2 tables then removes for anything nested deeper. */
        table {
            border-spacing: 0 !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            margin: 0 auto !important;
        }
        table table table {
            table-layout: auto;
        }

        /* What it does: Prevents Windows 10 Mail from underlining links despite inline CSS. Styles for underlined links should be inline. */
        a {
            text-decoration: none;
        }

        /* What it does: Uses a better rendering method when resizing images in IE. */
        img {
            -ms-interpolation-mode:bicubic;
        }

        /* What it does: A work-around for email clients meddling in triggered links. */
        *[x-apple-data-detectors],  /* iOS */
        .unstyle-auto-detected-links *,
        .aBn {
            border-bottom: 0 !important;
            cursor: default !important;
            color: inherit !important;
            text-decoration: none !important;
            font-size: inherit !important;
            font-family: inherit !important;
            font-weight: inherit !important;
            line-height: inherit !important;
        }

        /* What it does: Prevents Gmail from displaying a download button on large, non-linked images. */
        .a6S {
           display: none !important;
           opacity: 0.01 !important;
       }
       /* If the above doesn't work, add a .g-img class to any image in question. */
       img.g-img + div {
           display: none !important;
       }

        /* What it does: Removes right gutter in Gmail iOS app: https://github.com/TedGoas/Cerberus/issues/89  */
        /* Create one of these media queries for each additional viewport size you'd like to fix */

        /* iPhone 4, 4S, 5, 5S, 5C, and 5SE */
        @media only screen and (min-device-width: 320px) and (max-device-width: 374px) {
            .email-container {
                min-width: 320px !important;
            }
        }
        /* iPhone 6, 6S, 7, 8, and X */
        @media only screen and (min-device-width: 375px) and (max-device-width: 413px) {
            .email-container {
                min-width: 375px !important;
            }
        }
        /* iPhone 6+, 7+, and 8+ */
        @media only screen and (min-device-width: 414px) {
            .email-container {
                min-width: 414px !important;
            }
        }

    </style>
   
    <style>

        /* What it does: Hover styles for buttons */
        .button-td,
        .button-a {
            transition: all 100ms ease-in;
        }
        .button-td-primary:hover,
        .button-a-primary:hover {
            background: #555555 !important;
            border-color: #555555 !important;
        }

        /* Media Queries */
        @media screen and (max-width: 600px) {

            .email-container {
                width: 100% !important;
                margin: auto !important;
            }

            /* What it does: Forces elements to resize to the full width of their container. Useful for resizing images beyond their max-width. */
            .fluid {
                max-width: 100% !important;
                height: auto !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }

            /* What it does: Forces table cells into full-width rows. */
            .stack-column,
            .stack-column-center {
                display: block !important;
                width: 100% !important;
                max-width: 100% !important;
                direction: ltr !important;
            }
            /* And center justify these ones. */
            .stack-column-center {
                text-align: center !important;
            }

            /* What it does: Generic utility class for centering. Useful for images, buttons, and nested tables. */
            .center-on-narrow {
                text-align: center !important;
                display: block !important;
                margin-left: auto !important;
                margin-right: auto !important;
                float: none !important;
            }
            table.center-on-narrow {
                display: inline-block !important;
            }

            /* What it does: Adjust typography on small screens to improve readability */
            .email-container p {
                font-size: 17px !important;
            }
        }

    </style>
   

</head>

<body width="100%" style="margin: 0; padding: 0 !important; mso-line-height-rule: exactly; background-color: #222222;">
    <center style="width: 100%; background-color: #222222;">
   
        <div style="display: none; font-size: 1px; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden; mso-hide: all; font-family: sans-serif;">
            (Optional) This text will appear in the inbox preview, but not the email body. It can be used to supplement the email subject line or even summarize the email's contents. Extended text preheaders (~490 characters) seems like a better UX for anyone using a screenreader or voice-command apps like Siri to dictate the contents of an email. If this text is not included, email clients will automatically populate it using the text (including image alt text) at the start of the email's body.
        </div>
       
        <div style="display: none; font-size: 1px; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden; mso-hide: all; font-family: sans-serif;">
            &zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;
        </div>
      
        <table align="center" role="presentation" cellspacing="0" cellpadding="0" border="0" width="600" style="margin: 0 auto;" class="email-container">
           
            <tr>
                <td style="background-color: #ffffff;">
                    <center>
                        <img src="cid:image1" alt="build a model logo" style="width: 30%; max-width: 600px; height: auto; background: #dddddd; font-family: sans-serif; font-size: 15px; line-height: 15px; color: #555555; margin: auto;" class="g-img">
                    </center>
                </td>
            </tr>
           
            <tr>
                <td style="background-color: #ffffff;">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                        <tr>
                            
                                <td style="padding: 20px; font-family: sans-serif; font-size: 15px; line-height: 20px; color: #555555; text-align: center;">
                                    <h1 style="margin: 0 0 10px; font-size: 25px; line-height: 30px; color: #333333; font-weight: normal;">Hurray! You just completed a session with the Alexa skill "Build a model"</h1>
                                    <p style="margin: 0 0 10px; ">The identified core concept is - <i><b>"""+coreConcept+"""</b></i></p>
                                    <p style="margin: 0 0 10px;">Here is your generated causal map along with your answers - </p>
                                </td>
                        </tr>

                        <table style="margin-left: auto; margin-right: auto">
                            
                            <tr>
                                """ + htmlline + """
                            </tr>
                            
                        </table>

                        <tr>
                            <td style="background-color: #ffffff;">
                                <center>
                                    <img src="cid:image2" alt="Causal map" style="width: 60%; max-width: 600px; height: auto; background: #dddddd; font-family: sans-serif; font-size: 15px; line-height: 15px; color: #555555; margin: auto;" class="g-img">
                                    <p style="margin: 0 0 10px;font-size:12px"><u>The causal map and the csv files have been attached</u></p>
                                </center>

                            </td>
                        </tr>

                    </table>
                </td>
            </tr>
            
           

        </table>
       
   
    </center>
</body>
</html>
    """

    msg.attach(MIMEText(body,'html'))

    #attaching the images in the html template
    fp = open('logo.png', 'rb')
    logo_img = MIMEImage(fp.read())
    fp.close()
    logo_img.add_header('Content-ID', '<image1>')
    msg.attach(logo_img)

    fp = open('/tmp/networkx.png', 'rb')
    map_img = MIMEImage(fp.read())
    fp.close()
    map_img.add_header('Content-ID', '<image2>')
    msg.attach(map_img)
    #end of attaching the images in the html template




    #attaching download attachements to the email being sent
    attachment_a_file("/tmp/networkx.png")
    attachment_a_file("/tmp/node_relations.csv")

    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_user,email_password)
    server.sendmail(email_user,email_send,text)
    server.quit()

def send_log_to_the_developer_when_skill_breaks(session, error):
    session_1 = session['attributes']
    if(error == True):
        session_1['intent_triggered'].append("Error_has_occured")
        msg['Subject'] = 'Skill Broke - You are receiving this email because "Build a model" skill broke, See the attached log'
    body = str(session_1)
    
    email_user = email_config.EMAIL_ADDRESS
    email_password = email_config.PASSWORD
    email_send = 'datalab.science@gmail.com'

    subject = 'Thank you for using Build a Model - Here are your results'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = 'The session was completed successfully - here is the log'

    msg.attach(MIMEText(body,'plain'))

    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_user,email_password)
    server.sendmail(email_user,email_send,text)
    server.quit()

def on_session_started(session_started_request, session):
    """ Called when the session starts """

    print("on_session_started requestId=" + session_started_request['requestId']
          + ", sessionId=" + session['sessionId'])


def on_launch(launch_request, session):
    """ Called when the user launches the skill without specifying what they
    want
    """
    

    print("on_launch requestId=" + launch_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # Dispatch to your skill's launch
    return get_welcome_response()


def on_intent(intent_request, session):
    """ Called when the user specifies an intent for this skill """

    print("on_intent requestId=" + intent_request['requestId'] +
          ", sessionId=" + session['sessionId'])

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']

    # Dispatch to your skill's intent handlers
    if intent_name == "ModelCreatorIntent":
        return model_creator_intent(intent, session)    
    elif intent_name == "AMAZON.HelpIntent":
        return help_intent_request(intent, session) 
    elif intent_name == "AMAZON.YesIntent":
        return yes_intent_request(intent, session) 
    elif intent_name == "AMAZON.CancelIntent" or intent_name == "AMAZON.NoIntent":
        return handle_session_end_request(intent, session, proceed = False)
    elif intent_name == "AMAZON.StopIntent":
        return stop_intent_request(intent, session) 
    elif intent_name == "AMAZON.RepeatIntent":
        return repeat_intent_request(intent, session)
    elif intent_name == "AMAZON.FallbackIntent":
        return fall_back_intent(intent, session)
    elif intent_name == "ProceedIntent":
        return proceed_intent_request(intent, session)
    elif intent_name == "RemoveIntent":
        return remove_intent_request(intent, session)
    elif intent_name == "CaptureConceptIntent":
        return capture_core_concept_intent_request(intent, session)
    else:
        return error_intent(intent, session)


def on_session_ended(session_ended_request, session):    
    """ Called when the user ends the session.

    Is not called when the skill returns should_end_session=True
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # add cleanup logic here
    

# --------------- Main handler ------------------

def lambda_handler(event, context):
    """ Route the incoming request based on type (LaunchRequest, IntentRequest,
    etc.) The JSON body of the request is provided in the event parameter.
    """
    print("event.session.application.applicationId=" +
          event['session']['application']['applicationId'])
    
    # try:
    #     access_token = event['context']['user']['accessToken']
    #     print(access_token)
    # except:
    #     access_token = None
    #     print("access token is not found")
        

    """
    Uncomment this if statement and populate with your skill's application ID to
    prevent someone else from configuring a skill that sends requests to this
    function.
    """
    # if (event['session']['application']['applicationId'] !=
    #         "amzn1.echo-sdk-ams.app.[unique-value-here]"):
    #     raise ValueError("Invalid Application ID")

    if event['session']['new']:
        on_session_started({'requestId': event['request']['requestId']},
                           event['session'])
    if event['request']['type'] == "LaunchRequest":        
        return on_launch(event['request'], event['session'])

    elif event['request']['type'] == "IntentRequest":

        intent_triggered = event['session']['attributes']['intent_triggered']
        intent_triggered.append(event['request']['intent']['name'])        
        session_attributes.update({'intent_triggered':intent_triggered})

        return on_intent(event['request'], event['session'])


    elif event['request']['type'] == "SessionEndedRequest":

        intent_triggered = event['session']['attributes']['intent_triggered']
        intent_triggered.append(event['request']['intent']['name'])
        session_attributes.update({'intent_triggered':intent_triggered})

        return on_session_ended(event['request'], event['session'])
