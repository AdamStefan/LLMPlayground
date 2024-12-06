from  agents.llm_model import FunctionToolDefinition, ToolParameter
import agents.react.prompts.react_prompt as ReAct
from agents.react.agentic import ReActAgenticLoop
import  agents.agent as agent

def test_react():    
    args = {
    "deployment_name": "gpt-4o-2024-05-13", 
    "temperature": 0.0, 
    "max_tokens": 2000,
    "request_timeout": 60, 
    "model_kwargs": {"top_p": 0.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},        
    }    

    additional_info = """
Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
"""

    def calculate(what):
        return eval(what)
    
    def average_dog_weight(name):
        if name in "Scottish Terrier": 
            return("Scottish Terriers average 20 lbs")
        elif name in "Border Collie":
            return("a Border Collies average weight is 37 lbs")
        elif name in "Toy Poodle":
            return("a toy poodles average weight is 7 lbs")
        else:
            return("An average dog weights 50 lbs")

    
    calculate_tool = FunctionToolDefinition(name="calculate", 
                               description="Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary", 
                               example="calculate: 4 * 7 / 3",
                               parameters=[ToolParameter(name="expression", is_required=True)])
    
    average_dog_weight_tool = FunctionToolDefinition(name="average_dog_weight", 
                               description="returns average weight of a dog when given the breed", 
                               example="average_dog_weight: Collie",
                               parameters=[ToolParameter(name="breed", is_required=True)])
    

    calculate_action_description = ReAct.ReActActionDescription(name=calculate_tool["name"], example=calculate_tool["example"], description=calculate_tool["description"])
    average_weight_description = ReAct.ReActActionDescription(name=average_dog_weight_tool["name"], example=average_dog_weight_tool["example"], description=average_dog_weight_tool["description"])    

    react_system_prompt = ReAct.build_system_prompt(available_actions=[calculate_action_description, average_weight_description], additional_info=additional_info)
    
    react_agent = agent.Agent("react", react_system_prompt, llm_config=args, tools=None, default_examples=[])      
    
    loop = ReActAgenticLoop("react_agent_loop", [agent.AgentTool(calculate_action_description, calculate), agent.AgentTool(average_dog_weight_tool, average_dog_weight)], max_turns=5)
    out = loop.execute(react_agent,"How much does a toy poodle weigh?")
    print(out)


if __name__ == "__main__":
    test_react()
