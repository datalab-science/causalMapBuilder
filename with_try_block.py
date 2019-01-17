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
dynamoTable = dynamodb.Table('my_map')

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

WELCOME_MESSAGE ="Hi welcome to "+SKILL_NAME+"! We will be building a causal model for a problem of your choice. I will ask you which concepts drive this problem. Then, we will look at the drivers on these concepts. For more information regarding "+SKILL_NAME+" you can say 'Alexa Help me'. Shall we begin?"
HELP_MESSAGE = """"Let's """+SKILL_NAME+"""" is a skill that creates causal models out of your answers. 
    For example if you wish to make a causal model of obesity, questions will be asked such as "what do you think caues obesity?". 
    I will identify the main factors in your answers, and build on these to ask more questions. 
    Once all the questions are answered, I will create a causal map, and email to the registered email id with Amazon.
    In short, we're going to have a conversation, and I will track your thoughts by building a causal map. 
    Can we continue?"""

HELP_REPROMPT = "All you have to do is answer to my questions in a single word or a small phrase with your answers I will create causal map and email it to your registered email id with amazon. Ready to resume where you left off?"

STOP_MESSAGE = 'Thank you for your interest in using '+SKILL_NAME+'. Have a nice day!'

NO_OTHER_ENTITES = 'Thank you for answering all the questions. The causal map will be mailed to your email ID';
ERROR_MESSAGE ='There has occured an error';

BEGIN_MESSAGE = 'Ok, lets begin. ';
CAPTURE_ENTITY = 'We need a core concept for your map. For instance, you could create a map about obesity or homelessness. What would you like the core concept of your map to be?';


ROUND_2_QUESTION = ' Now can you please tell me all the causes of '
ROUND_2_QUESTION_FOLLOW_UP = 'Can you name all the causes of '
ROUND_2_QUESTION_FOLLOW_UP_ANOTHER = 'Can you name another cause of ';
ROUND_2_QUESTION_FOLLOW_UP_1 = ' Go ahead, Name all the causes for '
ROUND_2_ENTITIES_IDENTIFIED = 'You have provided all the causes of '

ROUND_2_DELETE_ENTITIES = ' If there are any causes that you\'d like me to remove, please say remove followed by the word. So far, we discussed causes of '
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

def and_string(entities_list):
    and_text = ''
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
    print(card_title)
    session_attributes.update({
        'yes_begin_skill':True, 
        'no_begin_skill':True,
        'synonyms':{}, 
        'question':[],
        'intent_triggered':[],
        'question_entities':[], 'answer':[], 
        'current_question_word':'',
        'stop_layer':False, 
        'help_intent':False
    })
    should_end_session = False    
    speech_output = WELCOME_MESSAGE   
    reprompt_text = WELCOME_MESSAGE
    session_attributes.update({'repeat':speech_output})
    question = []
    question.append("Welcome to "+SKILL_NAME+", Shall we begin?")
    session_attributes.update({'question':question})      
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))    

def handle_session_end_request(intent, session):  
     
    answer = session['attributes']['answer']
    answer.append("No")
    session_attributes.update({'answer': answer})
    
    question = session['attributes']['question']
    
    card_title = intent['name']

    try:
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
            identified_entites = session['attributes']['identified_entites']
            current_question_word = session['attributes']['current_question_word']
            print("Second round entered")
            CURRENT_LAYER_DEPTH = current_layer_depth_function(session)
            # if its the end of all the entities extracted
            #Prints the list of all the entities, builds a map, email's the map to the user, updates the log to dynamo db

            if(entities_count == len(first_layer_entities)-1):
                print("entered entities_count == len(first_layer_entities)")
                card_title = SKILL_NAME+" ended"
                session_attributes.update({
                    'another_entity_round_2': False,
                    'no_other_entities_round_2': False, 
                    'round_2':False
                })
                should_end_session = True
                and_string = ''

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

                    # speech_output = NO_OTHER_ENTITES+'I noted that the causes of '+ first_layer_entities[0] +' are '+first_layer_entities_string+" .Results and the causal map will be mailed to your email ID"
                    speech_output = NO_OTHER_ENTITES
                    reprompt_text = speech_output
                        
                    create_a_map(identified_entites)
                    send_email('tshivara@lakeheadu.ca')
                    update_into_table(session, error = False)
                    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

                

            #if its not the end of all the entities extracted then ask more questions 

            else:
                print("not the end of all the entities extracted then ask more questions")
                print("current_question_word="+current_question_word)
                print("identified_entites.keys()="+','.join(identified_entites.keys()))
                
                for x in identified_entites:
                    if(x == current_question_word):
                        print(x +"=="+ current_question_word)
                        if((len(identified_entites[x]) != 0) and (session['attributes']['no_more_entites'] == True)): 
                            CURRENT_LAYER_DEPTH = current_layer_depth_function(session)
                            print("entered len(identified_entites[{}]) != 0)".format(x)) 
                            should_end_session = False
                            #ask for the entites if they have to be removed
                            and_string = ''
                            if(len(identified_entites[current_question_word]) == 1):
                                for i in identified_entites[current_question_word]:
                                    and_string = i
                            else:
                                for i in range(0, len(identified_entites[current_question_word])-1):
                                    and_string+= identified_entites[current_question_word][i]+", "
                                and_string +=" and "+ identified_entites[current_question_word][len(identified_entites[current_question_word])-1]        

                            speech_output = ROUND_2_ENTITIES_IDENTIFIED + current_question_word +". I took note of the following causes: {} ".format(and_string)+"."+ROUND_2_DELETE_ENTITIES+first_layer_entities[0]+' up to a distance of ' +str(CURRENT_LAYER_DEPTH)+', If you don\'t want to discuss more distal causes, you can say "stop". To continue, please say "proceed".'
                            reprompt_text = speech_output 
                            if first_layer_entities[entities_count+1] not in identified_entites.keys():
                                for x in range(entities_count,len(first_layer_entities)-entities_count):
                                    print(first_layer_entities[x])
                                    if first_layer_entities[x] not in identified_entites.keys():
                                        identified_entites.update({first_layer_entities[x]:[]})    
                            # fetch_question_entites(reprompt_text, session)
                            question.append(reprompt_text)
                            print("about to enter current_question_word not in identified_entites")
                            session_attributes.update({
                                'question':question, 
                                'repeat':speech_output,
                                'round_1':False,
                                'round_2':True,
                                'identified_entites': identified_entites,
                                'current_question_word':first_layer_entities[entities_count],
                                'no_more_entites':False,
                                'entities_count': entities_count,
                                'stop_layer':True,
                                'no_more_entites':False
                            })
                            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

                        else:
                            print("entered len(identified_entites[{}]) == 0)".format(x))
                            current_question_word = first_layer_entities[entities_count]
                            entities_count = entities_count + 1
                            should_end_session = False
                            speech_output = ROUND_2_QUESTION_FOLLOW_UP + first_layer_entities[entities_count] + " that you're aware of ?"
                            reprompt_text = speech_output
                            # fetch_question_entites(reprompt_text, session)   
                            current_question_word = first_layer_entities[entities_count]
                            identified_entites[current_question_word] = []
                            identified_entites.update(identified_entites[current_question_word])
                            question.append(reprompt_text)
                            session_attributes.update({'entities_count':entities_count,
                                'round_2':True,
                                'no_more_entites':False,
                                'current_question_word':current_question_word,
                                'identified_entites':identified_entites,
                                'question':question,
                                'repeat':speech_output
                            })                   
                            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))    
  
    except:
        try_block_error(session)

def yes_intent_request(intent, session):
    
    answer = session['attributes']['answer']
    answer.append("Yes")
    session_attributes.update({'answer': answer})

   
    try:
        if(session['attributes']['help_intent'] == True):
            # if(len(session['attributes']['intent_triggered']) == 1):
                return get_welcome_response()
            # else:
                # previous_intent = session['attributes']['intent_triggered'][len(session['attributes']['intent_triggered'])-1]

                # if previous_intent == "AMAZON.HelpIntent":
                    # return get_welcome_response() 

                # if previous_intent == "ModelCreatorIntent":
                #     return model_creator_intent(intent, session)  
                # elif previous_intent == "AMAZON.YesIntent":
                #     return yes_intent_request(intent, session)
                #     session_attributes.update({'help_intent':False})
                # elif previous_intent == "AMAZON.CancelIntent" or previous_intent == "AMAZON.NoIntent":
                #     return handle_session_end_request(intent, session)
                # elif previous_intent == "AMAZON.StopIntent":
                #     return stop_intent_request(intent, session) 
                # elif previous_intent == "AMAZON.RepeatIntent":
                #     return repeat_intent_request(intent, session)
                # elif previous_intent == "AMAZON.FallbackIntent":
                #     return fall_back_intent(intent, session)
                # elif previous_intent == "ProceedIntent":
                #     return proceed_intent_request(intent, session)
                # elif previous_intent == "RemoveIntent":
                #     return remove_intent_request(intent, session)
                # elif previous_intent == "CaptureConceptIntent":
                #     return capture_core_concept_intent_request(intent, session)
                # else:
                #     return error_intent(intent, session)

                    

        question = session['attributes']['question']
        if(session['attributes']['yes_begin_skill'] == True):
            print("entered yes intent")
            card_title = intent['name']
            speech_output = BEGIN_MESSAGE + CAPTURE_ENTITY
            reprompt_text = CAPTURE_ENTITY 
            # fetch_question_entites(reprompt_text, session)
            question.append(reprompt_text)       
            should_end_session = False
            session_attributes.update({
                'yes_begin_skill':False, 
                'no_begin_skill':False, 
                'round_1':True,
                'entities_count':0, 
                'first_layer_entities':[],
                'identified_entites':{},
                'no_more_entites':True,
                'repeat':speech_output,
                'question':question,
                'stop_layer':False
            })
            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
        else:
            
            identified_entites = session['attributes']['identified_entites']
            print("entered yes I can name another entity intent")
            entities_count = session['attributes']['entities_count']
            first_layer_entities = session['attributes']['first_layer_entities']
            CURRENT_LAYER_DEPTH = current_layer_depth_function(session)
            card_title = intent['name']   
            speech_output = ROUND_2_QUESTION_FOLLOW_UP_1 + first_layer_entities[entities_count]
            reprompt_text = speech_output
            # fetch_question_entites(reprompt_text, session)   
            should_end_session = False
            question.append(reprompt_text)
            session_attributes.update({
                'question':question,
                'repeat':speech_output
            })
            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
    except:
       try_block_error(session)

def proceed_intent_request(intent, session):
    print("sending the pointer to no entity")
    return handle_session_end_request(intent, session)

def remove_intent_request(intent, session):
    print("entered remove intent")
    card_title = intent['name']

    should_end_session = False

    
    first_layer_entities = session['attributes']['first_layer_entities']
    entities_count = session['attributes']['entities_count']
    CURRENT_LAYER_DEPTH = current_layer_depth_function(session)  
    question_entities = session['attributes']['question_entities']
    current_question_word = session['attributes']['current_question_word']
    identified_entites = session['attributes']['identified_entites']
    answer = session['attributes']['answer']
    question = session['attributes']['question']

    try:

        if(len(current_question_word)>1):
            if((current_question_word in identified_entites.keys()) and len(identified_entites.keys()) > 0):
                identified_entites_list = identified_entites[current_question_word]
            else: identified_entites_list = []
        else: identified_entites_list = []
        
        current_question_word = first_layer_entities[entities_count]
        identified_entites_list = identified_entites[current_question_word]

        if(len(identified_entites_list) == 1):
            speech_output = "The "+ current_question_word+ "contains only one cause. Cannot delete all the causes, should contain at least one cause. Please say 'proceed' to go to the next step or 'stop' to end this session here"
            reprompt_text = speech_output
            question.append(reprompt_text)
            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
        else:
            answer_from_user = intent['slots']['remove_entity']['value']
            response = fetch_entities_from_google_api(answer_from_user)
            remove_entities_list = []
            for entity in response.entities:            
                if entity.name in first_layer_entities:
                    remove_entities_list.append(entity.name)
                    print("remove_entities_list:"+','.join(remove_entities_list))
                  
            for x in remove_entities_list:
                if x in first_layer_entities:
                    first_layer_entities.remove(x)
                    identified_entites_list.remove(x)
            # entities_count = entities_count + 1
            identified_entites[current_question_word] = identified_entites_list

            session_attributes.update({'identified_entites':identified_entites})
            CURRENT_LAYER_DEPTH = current_layer_depth_function(session)
            and_text = and_string(remove_entities_list)
            and_text_remaining = and_string(identified_entites[current_question_word])
            
            speech_output = and_text +" has been removed . The remaining causes for "+ current_question_word+ " include "+and_text_remaining+". If you would like to remove any other causes, Please say remove followed by the word, If not say 'proceed' to continue"
            reprompt_text = speech_output
            question.append(reprompt_text)
            session_attributes.update({
                'first_layer_entities':first_layer_entities,
                'entities_count':entities_count,
                'current_question_word':first_layer_entities[entities_count],
                'question':question,
                'repeat':speech_output            
            })

            return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

    except:
        try_block_error(session)

def model_creator_intent(intent, session):
    
    first_layer_entities = session['attributes']['first_layer_entities']
    entities_count = session['attributes']['entities_count']
    question_entities = session['attributes']['question_entities']
    current_question_word = session['attributes']['current_question_word']
    identified_entites = session['attributes']['identified_entites']
    answer = session['attributes']['answer']
    question = session['attributes']['question']
    synonyms = session['attributes']['synonyms']
    
    try:
        if(len(current_question_word)>1):
            if((current_question_word in identified_entites.keys()) and len(identified_entites.keys()) > 0):
                identified_entites_list = identified_entites[current_question_word]
            else:
                identified_entites_list = []
        else:
            identified_entites_list = []

        print("entered model_creator_intent_request")

        card_title = intent['name']
        
        should_end_session = False

        answer_from_user = intent['slots']['user_causes']['value']
        answer.append(answer_from_user)
        print("answer_from_user:"+answer_from_user)
        
        session_attributes.update({ 
            'no_begin_skill':False,
            'answer':answer        
        })
        word_list = answer_from_user.split()
        answer_from_user = ' '.join([i for i in word_list if i not in ["core","map","concept"]])
        #-------------------google API--------------------------#
        response = fetch_entities_from_google_api(answer_from_user)
        #-------------------google API--------------------------#

        entities_count = session['attributes']['entities_count']     
        
        for entity in response.entities:
            print('entity:',entity)
            #if the entity captured is a phrase with more than 2 words, split the words (this is done because alexa capture words without a comma when spoken fastly)
            # if (entity.name not in question_entities) and (entity.name not in first_layer_entities) and (entity.name not in syn_dict):
            if (entity.name not in first_layer_entities) and (entity.name not in syn_dict):
                print('entity.name',entity.name)
                if(len((entity.name).split()) > 2):
                    entities_split = (entity.name).split()
                    for x in entities_split:
                        first_layer_entities.append(x)
                        identified_entites_list.append(x)
                else:
                    first_layer_entities.append(entity.name)
                    identified_entites_list.append(entity.name)
                    print(identified_entites_list)

        #fecthes the synonyms and also checks if any entity fecthed is already captured or not
        fetch_synonyms(identified_entites_list,session)

        # if the core concept has not been captured yet
        if(len(question) == 2):
        # if the named core concept is more than a single concept
            if(len(identified_entites_list) == 1):
                print("entered len(identified_entites_list) == 1" )
                identified_entites.update({identified_entites_list[0]:[]})

                print("updated {} in identified_entites".format(identified_entites_list[0]))
                first_layer_entities = remove_duplicates(first_layer_entities) 
                speech_output = "Great, we'll make a map for "+identified_entites_list[0]+"."+ROUND_2_QUESTION + identified_entites_list[0] + " that you're aware of?"
                reprompt_text = speech_output
                question.append(reprompt_text)
                session_attributes.update({
                        'identified_entites': identified_entites,
                        'first_layer_entities':first_layer_entities,
                        'question':question,
                        'repeat':speech_output,
                        'current_question_word':identified_entites_list[0]
                    })
                
                return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
            
            else:
                
                first_layer_entities = remove_duplicates(first_layer_entities) 
                speech_output = "Please name just a single concept"
                reprompt_text = speech_output
                question.append(reprompt_text)
                session_attributes.update({
                        'repeat':speech_output
                    })
                return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

        # if the core concept has already been captured     
        identified_entites[current_question_word] = identified_entites_list
        first_layer_entities = remove_duplicates(first_layer_entities) 
        speech_output = ROUND_2_QUESTION_FOLLOW_UP_ANOTHER + first_layer_entities[entities_count] +" that you're aware of ?"
        reprompt_text = speech_output
        question.append(reprompt_text)
        session_attributes.update({
                'identified_entites': identified_entites,
                'first_layer_entities':first_layer_entities,
                'question':question,
                'repeat':speech_output,
                'no_more_entites':True
            })
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

    except:
       try_block_error(session)
        

def help_intent_request(intent, session):
    
    identified_entites = session['attributes']['identified_entites']

    try:
        
        print("entered help intent")
        card_title = intent['name']
        session_attributes.update({'yes_begin_skill':False, 'no_begin_skill':True, 'help_intent':True})
        should_end_session = False

        speech_output = HELP_MESSAGE
        reprompt_text = HELP_MESSAGE
        session_attributes.update({'repeat':speech_output})
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))
    except:
        try_block_error(session)

def fall_back_intent(intent, session):
    identified_entites = session['attributes']['identified_entites']
    card_title = intent['name']
    should_end_session = False
    speech_output = "Please answer in a proper sentence and not just words. Try using sentence starters to answer the questions"
    reprompt_text = speech_output
    send_log_to_the_developer_when_skill_breaks(session,error=True)
    create_a_map(identified_entites)
    send_email('tshivara@lakeheadu.ca')
    update_into_table(session, error = True)
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

def stop_intent_request(intent, session):
    
    answer = session['attributes']['answer']
    answer.append("Stop")
    session_attributes.update({'answer': answer})
    stop_layer = session['attributes']['stop_layer']
    
    entities_count =  session['attributes']['entities_count']
    CURRENT_LAYER_DEPTH = current_layer_depth_function(session)
    first_layer_entities = session['attributes']['first_layer_entities']
    identified_entites = session['attributes']['identified_entites']
    current_question_word = session['attributes']['current_question_word']   
    try:
        if(session['attributes']['stop_layer'] == True):
            print("Stop I do not want to enter next layer")

            # if its the end of all the entities extracted
            #Prints the list of all the entities, builds a map, email's the map to the user, updates the log to dynamo db

            card_title = SKILL_NAME+" ended"
            session_attributes.update({
                'another_entity_round_2': False,
                'no_other_entities_round_2': False, 
                'round_2':False
            })
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

                # speech_output = NO_OTHER_ENTITES+'I noted that the causes of '+ first_layer_entities[0] +' are '+first_layer_entities_string+" .Results and the causal map will be mailed to your email ID"
                speech_output = NO_OTHER_ENTITES
                reprompt_text = speech_output 
               
                create_a_map(identified_entites)
                send_email('tshivara@lakeheadu.ca')
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
    except:
       try_block_error(session)

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

def try_block_error(session):
    should_end_session = True
    identified_entites = session['attributes']['identified_entites']
    speech_output = "There has occured an error but the answers have been recorded and mailed to you"
    reprompt_text = speech_output
    send_log_to_the_developer_when_skill_breaks(session, error=True)  
    create_a_map(identified_entites)
    send_email('tshivara@lakeheadu.ca')
    update_into_table(session,error = True)
    return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))

def fetch_synonyms(words_list,session):
    synonyms = session['attributes']['synonyms']
    already_exisiting_entites = {}
    for word in words_list:
        syn_list = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas():
                for x in range(0,len(synonyms)):
                    if(l.name() == list(synonyms.keys())[x]):
                       already_exisiting_entites.update({l.name():list(synonyms.keys())[x]})
                    elif(l.name() in list(synonyms.values())[x]):
                        already_exisiting_entites.update({l.name():list(synonyms.values())[x]}) 
                if(l.name() not in list(synonyms.keys())):                      
                    syn_list.add(l.name())
        synonyms.update({word:list(syn_list)})

    session_attributes.update({'synonyms':synonyms})
    if(already_exisiting_entites):
        card_title = "Entites are repeated"
        relation = []
        for x in range(len(already_exisiting_entites)):
            relation.append([str(list(already_exisiting_entites.keys())[x] +" is a synonym of "+list(already_exisiting_entites.values())[x])])

        and_string = and_string(relation)
        should_end_session = False
        speech_output = "It looks like "+and_string+", hence these entites have not been captured or saved. Please say 'proceed' to continue or 'Stop' to stop here"
        reprompt_text = speech_output
        repeat = speech_output
        session_attributes.update({'repeat':repeat})
        return build_response(session_attributes, build_speechlet_response(card_title, speech_output, reprompt_text, should_end_session))


def current_layer_depth_function(session):
    current_question_word = session['attributes']['current_question_word']
    identified_entites = session['attributes']['identified_entites']

    for x in range(0,len(identified_entites)):
        if(current_question_word == list(identified_entites.keys())[x]):
            return x
            break
        elif(current_question_word in list(identified_entites.values())[x]):
            return x
            break

def fetch_entities_from_google_api(answer_from_user):

        document = language.types.Document(
            content=answer_from_user,
            language='en',
            type='PLAIN_TEXT',
        )
        response = client.analyze_entities(document=document, encoding_type='UTF32',)
        return response


def update_into_table(session, error):
    print("entered update_into_table")
    question = session['attributes']['question']

    intent_triggered = session['attributes']['intent_triggered']
    if(error == True):
        intent_triggered.append("error_has_occured")

    answer = session['attributes']['answer']
    identified_entites = session['attributes']['identified_entites']
    question_entities = session['attributes']['question_entities']

    dynamoTable.put_item(
            Item={
            'userId':session['sessionId'],
            'question':question,
            'intent_triggered':intent_triggered,
            'answer':answer,
            'identified_entites':identified_entites,
            'question_entities':question_entities
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
  
def create_a_map(identified_entites):
    plt.clf()
    fig_size = plt.rcParams["figure.figsize"]
    G=nx.DiGraph()
    csv_filename = '/tmp/node_relations.csv'
    with open(csv_filename, 'w') as f:
        wtr = csv.writer(f, delimiter=',')
        header = ["To","From"]
        wtr.writerow(i for i in header)
        
        #---------increase the image size -----------------#
        len_iden = len(identified_entites.keys())
        
        if(len_iden>=10 and len_iden<=20):
            plt.figure(figsize=(15,6))
        elif(len_iden>=20):
            print("entered len>20")
            plt.figure(figsize=(25,10))
        #---------increase the image size -----------------#

        for key in identified_entites.keys():       
            for z in range(0,len(identified_entites[key])):
                G.add_edges_from([(str(identified_entites[key][z]),str(key))])
                wtr.writerow([key,identified_entites[key][z]])
    f.close()
    nx.draw(G, with_labels=True, font_size=10, node_color='yellowgreen', pos=nx.spring_layout(G),  node_size=1000)
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
    
    body = str(session_1)
    
    email_user = email_config.EMAIL_ADDRESS
    email_password = email_config.PASSWORD
    email_send = 'tshivara@lakeheadu.ca'

    subject = 'Thank you for using Build a Model - Here are your results'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = 'Skill Broke - You are receiving this email because "Build a model" skill broke, See the attached log'

    msg.attach(MIMEText(body,'plain'))

    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_user,email_password)
    server.sendmail(email_user,email_send,text)
    server.quit()


# def fetch_question_entites(speech_text, session): 
#     question_entities = remove_duplicates(session['attributes']['question_entities'])
#     question = session['attributes']['question']   

#     document = language.types.Document(content=speech_text,language='en',type='PLAIN_TEXT',)
#     response = client.analyze_entities(document=document,encoding_type='UTF32',)

#     for entity in response.entities:
#         if (entity.name not in question_entities) and (entity.name not in syn_dict):
#                 question_entities.append(entity.name)
#     print("before:"+','.join(question_entities))
#     print(len(question))
#     if(len(question)==1):
#         print("entered len(question) ")
#         if('obesity' in question_entities):
#             question_entities.remove('obesity')
#             print("removed obesity")
#         elif('homelessness' in question_entities):
#             question_entities.remove('homelessness')
#             print("removed homelessness")
#     print("after:"+','.join(question_entities))

#     session_attributes.update({'question_entities':question_entities})


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
        return handle_session_end_request(intent, session)
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
