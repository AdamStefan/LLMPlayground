import agents.agent as agent
import re

class ReactAgentResultParser(object):
    def __init__(self):
        super().__init__()

    def parse_raw_content(self, raw_content)->agent.AgentStepResult:
        action_re = re.compile('^Action: (\w+): (.*)$')
        actions = [action_re.match(a)  for a in raw_content.split('\n') if action_re.match(a)]
        result =  agent.AgentStepResult(raw_content=raw_content, message=raw_content)

        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            result["actions"] = [(action, action_input)]

        return result                        
    
class ReActAgenticLoop(agent.AgenticLoop):
    def __init__(self, name:str, tools: list[agent.AgentTool], max_turns:int):
        super().__init__(name, tools, max_turns)
        self.result_parser = ReactAgentResultParser()                
    
    def execute(self, o_agent:agent.Agent, user_input:str):
        """
        Executes the custom tool
        """      
        i = 0    
        input =  user_input        
        while (i<self.max_turns):   
            agent_input = agent.AgentInput(user_message = input, functions_results = None)         
            result = o_agent.execute_step(agent_input)
            react_result = self.result_parser.parse_raw_content(result['raw_content'])
            observations:list[str] = []
            if 'actions' in react_result and len(react_result['actions']) > 0:
                for action_name, action_parameters in  react_result["actions"]:
                    if action_name in self.available_actions:
                        action_result = self.available_actions[action_name].execute_tool(action_parameters)
                        observations.append(action_result)

            if len(observations) == 0:
                return react_result['message']                                
            
            else:
                input = "Observation:" + "/n".join(observations)
            i = i+1
                  
        return react_result['message'] 
            

        
