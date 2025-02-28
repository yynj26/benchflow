import json
import logging
import os
from typing import Any, Dict

from openai import OpenAI
from benchflow import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TauBenchAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is missing")
        
        self.system_instruction = "You are an AI assistant helping a user with a customer service task. " \
                                  "Your job is to help the user accomplish their task by taking appropriate actions. " \
                                  "Always format your response as a valid JSON object with two fields: " \
                                  '"name": the name of the tool you want to use, "kwargs": a dictionary of parameters for the tool'

    def call_api(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        observation = env_info.get('observation', '')
        task = env_info.get('task', {})
        task_instruction = task.get('instruction', 'Help the user')
        tools_info = env_info.get('tools_info', [])
        
        message = f"Task: {task_instruction}\nObservation: {observation}\nAvailable Tools: {json.dumps(tools_info)}"
        
        client = OpenAI(api_key=self.api_key)
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": message}
                ],
                model="gpt-4o",
                temperature=0.7,
            )
            content = response.choices[0].message.content
            action = json.loads(content)
            
            if not isinstance(action, dict) or "name" not in action or "kwargs" not in action:
                raise ValueError("Invalid action")
                
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            action = {"name": "respond", "kwargs": {"content": f"Error: {str(e)}"}}
        
        return action

if __name__ == "__main__":
    print("Starting TauBench on port 9000...")
    agent = TauBenchAgent()
    agent.run_with_endpoint(host="0.0.0.0", port=9000)