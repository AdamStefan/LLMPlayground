from  agents.llm_model import LLMGatewayModel, ToolParameter, FunctionToolDefinition
import  agents.agent as agent
import json

args = {
    "deployment_name": "gpt-4o-2024-05-13", 
    "temperature": 0.0, 
    "max_tokens": 2000,
    "request_timeout": 60, 
    "model_kwargs": {"top_p": 0.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},        
    }


def test_llm_gateway():
    model = LLMGatewayModel(**args)

    image_path = r"C:\home\screen.jpg"
    #encoded_image_url = local_image_to_data_url(image_path)
    out = model.send_message([{"text":"You are an Ui Assistant expert that helps with describing screens","type":"system"},{"text":"Describe image","image_path":image_path,"type":"user"}])
    print(out)

def test_standard_agentic_loop():
    system_prompt = f'''
    Please extract the following information from the given text and return it as a JSON object:

    name
    major
    school
    grades
    club

    '''

    student_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."

    user_message = f''' This is the body of text to extract the information from:
    {student_description}'''


    def extract_student_info(name:str, major:str, school:str, grades:str, club:str):
        """Get the student information from the body of the input text"""
        return json.dumps({"name":name, "major":major ,"school":school, "grades":grades, "club":club})

    p1 = ToolParameter(name="name",description="Name of the person",is_required=True, type="string")
    p2 = ToolParameter(name="major",description="Major subject.",is_required=True, type="string")
    p3 = ToolParameter(name="school",description="The university name.",is_required=True, type="string")
    p4 = ToolParameter(name="grades",description="GPA of the student.",is_required=True, type="string")
    p5 = ToolParameter(name="club",description="School club for extracurricular activities.",is_required=True, type="string")

    function_tool = FunctionToolDefinition(name="extract_student_info",description="Get the student information from the body of the input text",parameters=[p1,p2,p3,p4,p5])


    a_agent = agent.Agent("standard_agent", system_prompt=system_prompt, tools=[function_tool],llm_config=args, default_examples=[])
    agentic_loop = agent.StandardFunctionCallingAgenticLoop("standar_agentic", tools=[agent.AgentTool(function_tool,tool=extract_student_info)], max_turns=2)

    out = agentic_loop.execute(a_agent,user_message)
    print(out)


if __name__ == "__main__":
    #test_llm_gateway()
    test_standard_agentic_loop()
   
    

    

