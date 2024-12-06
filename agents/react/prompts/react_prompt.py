
from agents.llm_model  import get_system_message, get_user_message, get_assistant_message
import typing_extensions as t
system_prompt_template = """
You are an expert assistant agent.
You run in a loop of Thought, Action, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the task you have been requested.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.

Your available actions are:

{available_actions_string}

""".strip()

action_description_template = """
{action_name}:
e.g. {action_call}
{action_description}
"""


class ReActAction(t.TypedDict):
    name:str
    observation:str
    thought:str


class ReActExample(t.TypedDict):
    query:str
    actions:list[ReActAction]

class ReActActionDescription(t.TypedDict):
    name:str
    description:str
    example:str


def build_examples_prompt(examples:list[ReActExample]):
    react_messages =  []
    for example in examples:
        query = example["query"]
        usr_message = get_user_message(query)
        example_actions = example["actions"]
        example_messages =  [usr_message]
        for i, action in enumerate(example_actions):
            action_messages = []
            is_last_action = (i  == len(example_actions) -1)
            thought = action["thought"]            
            action_value = action["value"]
            action_items = [f"Thought:{thought}"]
            if not is_last_action: 
                action_items.append(f"Action:{action_value}")               
                assistant_message = "/n".join(action_items)
                action_messages.append(assistant_message)
                observation = action["observation"] 
                action_messages.append(get_user_message(f"Observation:{observation}/n"))                               
            else:
                action_items.append(f"Answer:{action_value}")
                assistant_message = "/n".join(action_items)
                action_messages.append(assistant_message)
            example_messages.extend(action_messages)
        react_messages.extend(example_messages)
    return react_messages

def build_system_prompt(available_actions:list[ReActActionDescription], additional_info:str):  
    
    available_actions_text_list = []  

    for available_action in available_actions:
        action_description = action_description_template.format(action_name=available_action["name"], action_call = available_action["example"], action_description = available_action["description"])
        available_actions_text_list.append(action_description)

    available_actions_string = "/n/n".join(available_actions_text_list)

    system_message_prompt = system_prompt_template.format(available_actions_string = available_actions_string)

    if additional_info is not None and additional_info!= "":
        system_message_prompt = system_message_prompt + "/n/n" + additional_info    

    return system_message_prompt