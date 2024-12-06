import langchain.chat_models
import langchain
import langchain.prompts
from mimetypes import guess_type
import base64
import typing_extensions as t
import langchain_core.messages.tool as T
import langchain_core.messages as messages
import json




class FunctionCall(t.TypedDict):
    name:str
    parameters:dict[str, any]
    id:str

class ToolParameter(t.TypedDict):
    type:str
    name:str
    default_value:str
    description:str    
    is_required:bool

class FunctionToolDefinition(t.TypedDict):    
    name:str    
    description:str
    example: str
    parameters:list[ToolParameter]


class LLMMessage(t.TypedDict):
    type:str
    text:str

def get_user_message(text:str):
    return {"type":"user","text":text}

def get_system_message(text:str):
    return {"type":"system","text":text}

def get_assistant_message(text:str, tool_calls:list[FunctionCall]|None):
    out =  {"type":"assistant","text":text}

    if tool_calls is not None and len(tool_calls) > 0 :
        out["tool_calls"] = [ {"name" :tool_call["name"] , "id":tool_call["id"] , "args":tool_call["parameters"] }  for tool_call in tool_calls]
    return out

def get_tool_message(text:str, tool_id:str):
    return {"type":"tool","text":text, "tool_call_id":tool_id}

class LLMGatewayModel(langchain.chat_models.AzureChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def local_image_to_data_url(self, image_path):
        mime_type, _ = guess_type(image_path)
        # Default to png
        if mime_type is None:
            mime_type = 'image/png'

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"
    

    def bind_function_tools_dict(self, tools:list[FunctionToolDefinition]):
        function_tools = []
        for tool in tools:                
            tool_dict = {
                "name":tool["name"],
                "description":tool["description"],
                "parameters":{"type":"object","properties":{}}                
            }
            required_parameters = []

            for parameter in tool["parameters"]:
                tool_dict["parameters"]["properties"][parameter["name"]] = {"type":parameter["type"],"description":parameter["description"]}                
                if parameter["is_required"]:
                    required_parameters.append(parameter["name"])

            if len(required_parameters)>0:
                tool_dict["required"] = required_parameters
            
            function_tools.append(tool_dict)

        if len(function_tools) > 0:
            return self.bind(functions = function_tools)                        
        return self        


    def bind_tools_dict(self, tools:list[FunctionToolDefinition]):
        tools_calls = []
        for tool in tools:
            required_parameters = []                
            tool_dict = {                        
                        "type":"function",
                        "function":{
                            "name":tool["name"],
                            "description":tool["description"],
                            "parameters":{                                
                                "type":"object","properties":{}
                                }
                            }                                                                                                                
                    }
            properties_dict = tool_dict["function"]["parameters"]["properties"]
            
            for parameter in tool["parameters"]:
                properties_dict[parameter["name"]] = {"type":parameter["type"],"description":parameter["description"]}                
                if parameter["is_required"]:
                    required_parameters.append(parameter["name"])

            if len(required_parameters)>0:
                tool_dict["function"]["parameters"]["required"] = required_parameters
            tools_calls.append(tool_dict)
        if len(tools_calls) > 0:
            return self.bind(tools = tools_calls)                        
        return self
    
    def send_message(self, messages_items, tools:list[FunctionToolDefinition] = None, enforced_functions:dict[str,str] = None, parameters_dict = None):
        llm_messages = []
        if parameters_dict is None:
            parameters_dict = {}                                           
        message_index = 0    
            
        for message_index, message in enumerate(messages_items):        
            llm_message = None
            if message["type"] == "system":
                llm_message = langchain.prompts.SystemMessagePromptTemplate.from_template(message["text"])            
            elif message["type"] == "user":
                msg_template = []            
                if "text" in message:
                    msg_template.append({"type": "text", "text": message["text"]})                
                if "image_path" in message:
                    encoded_image_url = self.local_image_to_data_url(message["image_path"])                                
                    parameter_name = f"encoded_image_url_{message_index}"
                    parameters_dict[parameter_name] = encoded_image_url                
                    msg_template.append({"type": "image_url",  "image_url": f"{{{parameter_name}}}"})
                                    
                if len(msg_template)>0:                
                    llm_message = langchain.prompts.HumanMessagePromptTemplate.from_template(template = msg_template)                    
            elif message["type"] == "assistant":                                
                if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"])>0:
                    tool_calls_additional_args = []
                    tool_calls = []
                    for tool_call in message["tool_calls"]:
                        tool_calls.append(T.ToolCall(name = tool_call["name"], id = tool_call["id"], args = tool_call["args"]))                    
                        tool_call_dict = {  'id':tool_call["id"],
                                            'type':'function' ,
                                            'function': {
                                                'arguments': json.dumps(tool_call["args"]), 
                                                'name':tool_call["name"]
                                                        } 
                                        }
                        tool_calls_additional_args.append(tool_call_dict)
                    llm_message = messages.AIMessage(content = message["text"], tool_calls=tool_calls, additional_kwargs= {'tool_calls':tool_calls_additional_args})
                    
                else:
                    llm_message = messages.AIMessage(content=message["text"])                    
            elif message["type"] == "tool":
                llm_message = T.ToolMessage(content=message["text"],tool_call_id = message["tool_call_id"])

            if llm_message is not None:
                llm_messages.append(llm_message)
        chat_prompt = langchain.prompts.ChatPromptTemplate.from_messages(llm_messages)
        model  = self

        use_tools_call = True

        if tools is not None and len(tools)>0:            
            model = self.bind_tools_dict(tools) if use_tools_call else self.bind_function_tools_dict(tools)           

        chain = chat_prompt | model        
        out = chain.invoke(parameters_dict)
        
        functions_calls =  []
        if not use_tools_call:        
            if 'function_call' in out.additional_kwargs:
                function_call = out.additional_kwargs["function_call"]
                function_name = function_call["name"]
                function_item = FunctionCall(name=function_name,parameters=function_call["arguments"],id = out.id)
                functions_calls.append(function_item)
        else:
            for tool in out.tool_calls:
                function_name = tool["name"]
                id = tool["id"]
                function_item = FunctionCall(name=function_name, parameters=tool["args"], id=id)                
                functions_calls.append(function_item)
        
        return {"content":out.content, "functions_calls" : functions_calls}
    

if __name__ == "__main__":
    

    from  agents.llm_model import LLMGatewayModel
    from pydantic import BaseModel, Field
    from langchain_core.utils.function_calling import convert_to_openai_function
    from llm_model import FunctionToolDefinition, ToolParameter


    student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."
    prompt1 = f'''
    Please extract the following information from the given text and return it as a JSON object:

    name
    major
    school
    grades
    club

    This is the body of text to extract the information from:
    {student_1_description}
    '''

    class extract_student_info(BaseModel):
        """Get the student information from the body of the input text"""
        name: str = Field(description="Name of the person")
        major: str = Field(description="Major subject.")
        school: str = Field(description="The university name.")
        grades: str = Field(description="GPA of the student.")
        club: str = Field(description="School club for extracurricular activities.")

    p1 = ToolParameter(name="name",description="Name of the person",is_required=True, type="string")
    p2 = ToolParameter(name="major",description="Major subject.",is_required=True, type="string")
    p3 = ToolParameter(name="school",description="The university name.",is_required=True, type="string")
    p4 = ToolParameter(name="grades",description="GPA of the student.",is_required=True, type="string")
    p5 = ToolParameter(name="club",description="School club for extracurricular activities.",is_required=True, type="string")

    function_tool = FunctionToolDefinition(name="extract_student_info",description="Get the student information from the body of the input text",parameters=[p1,p2,p3,p4,p5])

    # extract_student_info_function = convert_to_openai_function(extract_student_info)

    args = {
        "deployment_name": "gpt-4o-2024-05-13", 
        "temperature": 0.0, 
        "max_tokens": 2000,
        "request_timeout": 60, 
        "model_kwargs": {"top_p": 0.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},        
        }

    model = LLMGatewayModel(**args)

    #encoded_image_url = local_image_to_data_url(image_path)
    out = model.send_message(messages_items = [{"text":prompt1,"type":"user"}], tools = [function_tool], enforced_functions={"name":"extract_student_info"})
    print(out)

