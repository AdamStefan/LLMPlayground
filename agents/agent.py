from  agents.llm_model import LLMGatewayModel, LLMMessage, get_system_message, get_user_message, get_tool_message, get_assistant_message, FunctionToolDefinition, ToolParameter, FunctionCall
import agents.react.prompts.react_prompt as ReAct
import typing_extensions as t
import abc
import json

class Memory(object):
    def __init__(self, system_prompt:str):
        self._chat_history = []
        self._system_prompt = system_prompt        

    def add_chat_message(self, message):
        self._chat_history.append(message)

    def get_chat_history(self):
        return self._chat_history    

class AgentTool(object):
    def __init__(self, definition:FunctionToolDefinition, tool):
        self.definition = definition
        self.tool = tool
    
    def execute_tool(self, *args, **kwargs)->str:
        """
        Method which executes the tool                
        """               
        return self.tool(*args, **kwargs)

class AgentStepResult(t.TypedDict):
    raw_content:str
    actions:list[FunctionCall]
    message:str


class AgentStepResultParser(object):
    def __init__(self):
        pass      

    @abc.abstractmethod
    def parse_raw_content(self, raw_content)->AgentStepResult:
        pass

class FunctionResult(t.TypedDict):
    function_id:str
    message:str

class AgentInput(t.TypedDict):
    user_message:str|None
    functions_results:list[FunctionResult]|None

        
class Agent():
    def __init__(self, name:str, 
                 system_prompt:str, 
                 tools:list[FunctionToolDefinition], 
                 llm_config:dict, 
                 default_examples:list[LLMMessage]):
        self.name =  name
        self.memory = Memory(system_prompt=system_prompt)
        self.system_prompt = system_prompt 
        self.tools = tools
        self.llm_config = llm_config
        self.default_examples = default_examples        

    def execute_step(self, input:AgentInput)->AgentStepResult:
        llm_wrapper = LLMGatewayModel(**self.llm_config)                
        messages = self.memory.get_chat_history()
        input_messages = []
        user_message = None

        if input["user_message"] is not None:
            user_message = get_user_message(input["user_message"])
            input_messages.append(user_message)
        elif input["functions_results"] is not None:
            for function_result in input["functions_results"]:
                function_message= get_tool_message(function_result["message"], function_result["function_id"])
                input_messages.append(function_message)                
                        
        initial_messages = [get_system_message(self.system_prompt)]
        if self.default_examples is not None and len(self.default_examples)>0:
            initial_messages = initial_messages + self.default_examples        

        out = llm_wrapper.send_message(initial_messages + messages + input_messages, tools=self.tools)

        for message in input_messages:
            self.memory.add_chat_message(message)
        assistant_message = get_assistant_message(out["content"], out["functions_calls"])
        self.memory.add_chat_message(assistant_message)        

        return AgentStepResult(raw_content=out["content"], message=out["content"], actions=out["functions_calls"]) 
                                                                                                           

class AgenticLoop(object):
    def __init__(self, name:str, tools: list[AgentTool], max_turns:int):
        self.name = name        
        self.max_turns = max_turns
        self.available_actions = {tool.definition["name"]: tool for tool in tools}


    @abc.abstractmethod
    def execute(agent:Agent, user_input:str):
         """
        Executes the custom tool
        """

class StandardFunctionCallingAgenticLoop(AgenticLoop):
    def __init__(self, name:str, tools: list[AgentTool], max_turns:int):
        super().__init__(name, tools, max_turns)

    def execute(self, agent:Agent, user_input:str):
        i = 0
        agent_input = AgentInput(user_message=user_input, functions_results=None)
        while(i<self.max_turns):            
            out = agent.execute_step(agent_input)
            if len(out["actions"]) > 0:
                functions_results:list[FunctionResult] = []
                for action in out["actions"]:
                    action_name = action["name"]                    
                    if action_name in self.available_actions:
                        action_parameters = action["parameters"]
                        result = self.available_actions[action_name].execute_tool(**action_parameters)
                        result_message_dict = {"function_name":action_name, "parameters":action_parameters, "result":result} 
                        result_message = json.dumps(result_message_dict)
                        functions_results.append(FunctionResult(function_id=action["id"],message=result_message))
                agent_input = AgentInput(user_message=None,functions_results=functions_results)
                if agent_input["functions_results"] is None or len(agent_input["functions_results"]) == 0:
                    return out["raw_content"]
            else:
                return out["raw_content"]
        return out["raw_content"]
        